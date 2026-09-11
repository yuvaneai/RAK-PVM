
import datetime
import os
import ast  # 安全解析字符串为字典
from argparse import ArgumentParser
import torch.nn.functional as F
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from dataset.data_module import DataModule
from dataset.pretrain_dataset import CXRMultiLabelPretrainingDataset
from dataset.transforms import DataTransforms
#
from backbones.image_encoder import Vit_ImageEncoder
from backbones.text_encoder import BertEncoder

from torch import distributed as dist

from backbones.anatomy_aware_prompt_generation import Seg_Prompt_Learner
from backbones.anatomy_patch_relevant import AnatomyTeacher, DensePromptForTest, AnatomyStudent

# 设置复现
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SAM_PROMPT(LightningModule):
    def __init__(self,
                 # 图像编码器参数
                 img_encoder: str = "vit_base",
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.img_encoder_q = Vit_ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)

        self.anatomy_box_T = AnatomyTeacher(INIT_EMBED_VOCAB=12520, POS_EMBED_DIM=self.hparams.trans_emb_dim)

        self.anatomy_box_S = AnatomyStudent(
            patch_dim=self.hparams.trans_emb_dim,
            hidden_dim=128,
            num_queries=30
        )

        self.prompt_learner = Seg_Prompt_Learner(transformer_dim=self.hparams.trans_emb_dim)

        pretrain_ckpt = os.path.join(BASE_DIR, f"./result/pretrain/ckpts/example.ckpt")

        our_state_dict = torch.load(
            pretrain_ckpt,
            map_location=torch.device("cpu"))

        our_state_dict = our_state_dict["state_dict"]

        img_encoder_state_dict = {k.replace("img_encoder_q.", ""): v  for k, v in our_state_dict.items() if k.startswith("img_encoder_q.")}
        prompt_learner_state_dict = {k.replace("prompt_learner.", ""): v for k, v in our_state_dict.items() if k.startswith("prompt_learner.")}
        anatomy_box_state_dict = {k.replace("anatomy_box_T.", ""): v for k, v in our_state_dict.items() if k.startswith("anatomy_box_T.")}

        self.anatomy_box_T.load_state_dict(anatomy_box_state_dict, strict=False)
        self.prompt_learner.load_state_dict(prompt_learner_state_dict, strict=False)
        self.img_encoder_q.load_state_dict(img_encoder_state_dict, strict=False)

        for param in self.prompt_learner.parameters():
            param.requires_grad = False

        for param in self.img_encoder_q.parameters():
            param.requires_grad = False

        for param in self.anatomy_box_T.parameters():
            param.requires_grad = False

    def forward(self, batch, batch_idx, split="train"):
        image = batch["image"]
        position = batch["position"]

        image_embeds = self.img_encoder_q.patch_embeddings(image)
        image_embeds = self.prompt_learner.img_proj(image_embeds)

        self.anatomy_box_T.eval()
        with torch.no_grad():
            best_patch_teacher, patch_comp_teacher, sim_matrix_teacher, mask = self.anatomy_box_T(
                patch_batch=image_embeds,
                pos_list_batch=position
            )

        student_out = self.anatomy_box_S(image_embeds)

        B_T, N_T, D_T = best_patch_teacher.shape
        B_S, N_S, D_S = student_out.shape
        N = min(N_T, N_S)
        student = student_out[:, :N, :]
        teacher = best_patch_teacher[:, :N, :]
        mask = mask[:, :N]
        mask = mask.squeeze(-1)  # [B, N]

        s_list, t_list = [], []
        student_sim_masked, teacher_sim_masked = [], []


        student_sim = torch.bmm(student, image_embeds.transpose(1, 2))  # [B, N, P]
        teacher_sim = sim_matrix_teacher[:, :N, :].detach()  # [B, N, P]

        for b in range(B_T):
            valid_idx = mask[b]  # [N] bool
            if valid_idx.sum() == 0:
                continue

            s_list.append(student[b, valid_idx, :])
            t_list.append(teacher[b, valid_idx, :])

            student_sim_masked.append(F.log_softmax(student_sim[b, valid_idx, :], dim=-1))
            teacher_sim_masked.append(F.softmax(teacher_sim[b, valid_idx, :], dim=-1))

        if len(s_list) == 0:

            raw_loss_cos = torch.tensor(0.0, device=student.device)
            raw_loss_l1 = torch.tensor(0.0, device=student.device)
            raw_loss_kl = torch.tensor(0.0, device=student.device)
        else:
            s_masked = torch.cat(s_list, dim=0)
            t_masked = torch.cat(t_list, dim=0)

            s_norm = F.normalize(s_masked, dim=-1)
            t_norm = F.normalize(t_masked.detach(), dim=-1)
            raw_loss_cos = 1 - F.cosine_similarity(s_norm, t_norm, dim=-1).mean()

            raw_loss_l1 = F.smooth_l1_loss(s_masked, t_masked.detach())

            p_student_all = torch.cat(student_sim_masked, dim=0)
            p_teacher_all = torch.cat(teacher_sim_masked, dim=0)
            raw_loss_kl = F.kl_div(p_student_all, p_teacher_all, reduction="batchmean")

        loss_cos = raw_loss_cos / (raw_loss_cos.detach() + 1e-8)
        loss_l1 = raw_loss_l1 / (raw_loss_l1.detach() + 1e-8)
        loss_kl = raw_loss_kl / (raw_loss_kl.detach() + 1e-8)

        return (loss_cos, loss_l1, loss_kl), (raw_loss_cos, raw_loss_l1, raw_loss_kl)


    def training_step(self, batch, batch_idx):
        (loss_cos, loss_l1, loss_kl), (raw_cos, raw_l1, raw_kl) = self(batch, batch_idx, "train")
        loss =  self.hparams.lambda_1 * loss_cos + self.hparams.lambda_2 * loss_l1 + self.hparams.lambda_3 * loss_kl
        raw_val = 1.0 * raw_cos  +  1.0 * raw_l1  +  0.1 * raw_kl  # 用来log、看曲线
        self.log_dict({"train_loss": loss},
                      sync_dist=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log_dict({"train_raw_loss": raw_val},
                      sync_dist=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        (loss_cos, loss_l1, loss_kl), (raw_cos, raw_l1, raw_kl) = self(batch, batch_idx, "valid")
        loss =  self.hparams.lambda_1 * loss_cos + self.hparams.lambda_2 * loss_l1 + self.hparams.lambda_3 * loss_kl
        raw_val = 1.0 * raw_cos  +  1.0 * raw_l1  +  0.1 * raw_kl  # 用来log、看曲线
        self.log_dict({"val_loss": loss},
                      sync_dist=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log_dict({"val_raw_loss": raw_val},
                      sync_dist=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def get_trainable_params(self):
        return [(k, v) for k, v in self.named_parameters() if v.requires_grad]

    def configure_optimizers(self):
        trainable_params = [p[1] for p in self.get_trainable_params()]
        optimizer = torch.optim.AdamW(trainable_params, self.hparams.learning_rate,
                                      betas=(self.hparams.momentum, 0.999),
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=int(self.training_steps * 0.25),
            cycle_mult=2.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-5,
            warmup_steps=int(self.training_steps * 0.05)
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", default=True, action="store_true")
        parser.add_argument("--freeze_img", default=True, action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=256, help="128, 256")
        parser.add_argument("--trans_emb_dim", type=int,
                            default=256, help="64, 128, 256, 512, 768")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.3)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1)
        parser.add_argument("--lambda_2", type=float, default=1)
        parser.add_argument("--lambda_3", type=float, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        return parser



    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_devices)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * trainer.max_epochs


def cli_main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = SAM_PROMPT.add_model_specific_args(parser)
    args = parser.parse_args()

    args.accelerator = "gpu"
    args.devices = 1
    args.strategy = "auto"
    args.gpus = None
    args.deterministic = True
    args.max_epochs = 50

    seed_everything(args.seed)

    datamodule = DataModule(CXRMultiLabelPretrainingDataset, DataTransforms,
                            args.data_pct, args.batch_size, args.num_workers)

    model = SAM_PROMPT(**args.__dict__)

    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"./result/distill/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger_dir = os.path.join(BASE_DIR, f"./result/distill/log")
    os.makedirs(logger_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name=extension)


    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_raw_loss", dirpath=ckpt_dir, save_top_k=5, save_last=True, mode="min"),
    ]

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=tb_logger,
        precision=32,
        amp_backend='native',
        benchmark=True,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )


    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(f"Total training steps: {model.training_steps}")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
