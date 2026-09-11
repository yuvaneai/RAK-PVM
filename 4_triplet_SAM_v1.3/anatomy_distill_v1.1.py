
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

from backbones.seg_prompt_learner import Seg_Prompt_Learner
from backbones.anatomy_patch_relevant import AnatomyTeacher, DensePromptForTest, AnatomyStudent

# 设置复现
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SAM_PROMPT(LightningModule):
    def __init__(self,
                 # 图像编码器参数
                 img_encoder: str = "vit_base",
                 emb_dim: int = 128,
                 # 文本编码器参数
                 freeze_bert: bool = False,
                 # transfomer_decoder维度
                 trans_emb_dim: int = 256,
                 # 局部对齐损失参数
                 local_temperature: float = 0.1,
                 bidirectional: bool = True,
                 num_prototypes: int = 50,
                 # 损失融合参数
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 proto_temperature: float = 0.2,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.img_encoder_q = Vit_ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)

        self.anatomy_box_T = AnatomyTeacher(INIT_EMBED_VOCAB=12520, POS_EMBED_DIM=self.hparams.trans_emb_dim)

        # 🔥 关键：num_queries 必须 ≥ 你教师最大解剖数量（50 就够）
        self.anatomy_box_S = AnatomyStudent(
            patch_dim=self.hparams.trans_emb_dim,
            hidden_dim=128,
            num_queries=30  # 必须 ≥ 教师的解剖位置数  学生 num_queries 只要 ≥ 教师一张图最多出现的解剖位置数量，就永远没问题！
        )

        self.prompt_learner = Seg_Prompt_Learner(transformer_dim=self.hparams.trans_emb_dim)
        # 256
        our_state_dict = torch.load(
            "/home/by/by/4_SAM_prompt/4_triplet_SAM_v1.3/result/pretrain_loss_ita_256token/ckpts/2026_04_10_16_37_27/epoch=30-step=14167.ckpt",
            map_location=torch.device("cpu"))
        # 128
        # our_state_dict = torch.load(
        #     "/home/by/by/4_SAM_prompt/4_triplet_SAM_v1.3/result/pretrain_loss_ita_128token/ckpts/2026_04_10_16_35_35/epoch=30-step=14167.ckpt",
        #     map_location=torch.device("cpu"))
        our_state_dict = our_state_dict["state_dict"]

        img_encoder_state_dict = {k.replace("img_encoder_q.", ""): v  for k, v in our_state_dict.items() if k.startswith("img_encoder_q.")}
        prompt_learner_state_dict = {k.replace("prompt_learner.", ""): v for k, v in our_state_dict.items() if k.startswith("prompt_learner.")}
        anatomy_box_state_dict = {k.replace("anatomy_box_T.", ""): v for k, v in our_state_dict.items() if k.startswith("anatomy_box_T.")}

        # 加载模型
        self.anatomy_box_T.load_state_dict(anatomy_box_state_dict, strict=False)
        self.prompt_learner.load_state_dict(prompt_learner_state_dict, strict=False)
        self.img_encoder_q.load_state_dict(img_encoder_state_dict, strict=False)

        for param in self.prompt_learner.parameters():
            param.requires_grad = False  # False

        for param in self.img_encoder_q.parameters():
            param.requires_grad = False  # False

        for param in self.anatomy_box_T.parameters():
            param.requires_grad = False  # False  # True


    def forward(self, batch, batch_idx, split="train"):
        image = batch["image"]
        position = batch["position"]

        # ===== 1. image encoder =====
        image_embeds = self.img_encoder_q.patch_embeddings(image)
        image_embeds = self.prompt_learner.img_proj(image_embeds)

        # ===== 2. teacher（冻结）=====
        self.anatomy_box_T.eval()
        with torch.no_grad():
            best_patch_teacher, patch_comp_teacher, sim_matrix_teacher, mask = self.anatomy_box_T(
                patch_batch=image_embeds,
                pos_list_batch=position
            )

        # ===== 3. student =====
        # student_out = self.anatomy_box_S(image_embeds)
        student_out, student_attn = self.anatomy_box_S(image_embeds)
        # ===== 4. 对齐 =====
        B, N, D = best_patch_teacher.shape
        student_aligned = student_out[:, :N, :]
        student_attn_aligned = student_attn[:, :N, :]

        mask = mask.squeeze(-1).bool()
        mask = mask[:, :student_attn_aligned.size(1)]

        # ============================
        # 1️⃣ Feature Alignment（cosine）  （学特征方向）
        # ============================
        s_feat = F.normalize(student_aligned[mask], dim=-1)
        t_feat = F.normalize(best_patch_teacher[mask].detach(), dim=-1)
        if mask.sum() < 5:
            loss_cos = torch.tensor(0.0, device=student_aligned.device)
        else:
            loss_cos = 1 - F.cosine_similarity(s_feat, t_feat, dim=-1).mean()

        # ============================
        # 2️⃣ Distribution Matching（KL）
        # ============================
        T = 0.3

        loss_attn = F.kl_div(
            F.log_softmax(student_attn_aligned / T, dim=-1),
            F.softmax(patch_comp_teacher.detach() / T, dim=-1),
            reduction="mean"
        )

        # ============================
        # 3️⃣ Global Consistency（新增🔥）
        # ============================
        student_global = student_aligned.mean(dim=1)
        teacher_global = best_patch_teacher.mean(dim=1)

        loss_global = 1 - F.cosine_similarity(student_global, teacher_global, dim=-1).mean()

        # # =========================
        # # 7. LOSS 4: Query diversity (防 collapse)
        # # =========================
        # q = self.anatomy_box_S.query_content + self.anatomy_box_S.query_pos
        # q = F.normalize(q, dim=-1)
        #
        # sim = torch.bmm(q, q.transpose(1, 2))
        # I = torch.eye(sim.size(-1), device=sim.device).unsqueeze(0)
        #
        # loss_div = ((sim - I) ** 2).mean()

        return loss_cos, loss_attn, loss_global


    def training_step(self, batch, batch_idx):
        loss_cos, loss_attn, loss_global= self(batch, batch_idx, "train")
        loss =  self.hparams.lambda_1 * loss_cos + self.hparams.lambda_2 * loss_attn + self.hparams.lambda_3 * loss_global
        # raw_val = 1.0 * raw_cos  +  1.0 * raw_l1  +  0.1 * raw_kl  # 用来log、看曲线
        self.log_dict({"train_loss": loss},
                      sync_dist=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_cos, loss_attn, loss_global = self(batch, batch_idx, "valid")
        loss =  self.hparams.lambda_1 * loss_cos + self.hparams.lambda_2 * loss_attn + self.hparams.lambda_3 * loss_global
        # raw_val = 1.0 * raw_cos  +  1.0 * raw_l1  +  0.1 * raw_kl  # 用来log、看曲线
        self.log_dict({"val_loss": loss},
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
                            default=256, help="64, 128, 256, 512, 768")  # 192
        parser.add_argument("--num_workers", type=int, default=16)  # 16
        parser.add_argument("--softmax_temperature", type=float, default=0.3)
        parser.add_argument("--learning_rate", type=float, default=1e-4)  # 1e-3   1e-4
        parser.add_argument("--momentum", type=float, default=0.9)  # ok
        parser.add_argument("--weight_decay", type=float, default=0.01)  # 以前0.05  过拟合才考虑增大
        parser.add_argument("--batch_size", type=int, default=128)   # 48 64
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1)
        parser.add_argument("--lambda_2", type=float, default=0.5)
        parser.add_argument("--lambda_3", type=float, default=0.5)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        return parser



    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        # num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        num_devices = max(1, trainer.num_devices)
        # if trainer.tpu_cores:
        #     num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * trainer.max_epochs


def cli_main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = SAM_PROMPT.add_model_specific_args(parser)
    args = parser.parse_args()

    # args.gpus = 2
    # args.strategy = "ddp"  #  "auto"ddp
    # args.deterministic = True
    # args.max_epochs = 50
    # args.max_epochs = 50

    args.accelerator = "gpu"
    args.devices = 1  # 你有几张卡写几
    args.strategy = "auto"  # 多卡必须用 ddp，不能用 auto
    args.gpus = None  # 删掉 gpus 参数，用 devices
    args.deterministic = True
    args.max_epochs = 50

    seed_everything(args.seed)

    datamodule = DataModule(CXRMultiLabelPretrainingDataset, DataTransforms,
                            args.data_pct, args.batch_size, args.num_workers)

    model = SAM_PROMPT(**args.__dict__)

    # checkpoint 保存目录
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"result/anatomy_distill_256token_studim128_v1.1/ckpts/{extension}")  # loss_ita+loss_proto
    os.makedirs(ckpt_dir, exist_ok=True)

    # TensorBoard 日志目录
    logger_dir = os.path.join(BASE_DIR, f"result/anatomy_distill_256token_studim128_v1.1/log")  # loss_ita+loss_proto
    os.makedirs(logger_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name=extension)


    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir, save_top_k=5, save_last=True, mode="min"),
        # EarlyStopping(monitor="val_loss", patience=5, verbose=False, mode="min")
    ]

    # trainer = Trainer.from_argparse_args(args, callbacks=callbacks, logger=tb_logger)
    # Lightning Trainer
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=tb_logger,
        precision=32,  # FP16
        amp_backend='native',
        benchmark=True,  # CuDNN 优化
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )


    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(f"Total training steps: {model.training_steps}")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    trainer.fit(model, datamodule=datamodule)

    # 保存 checkpoint 列表
    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
