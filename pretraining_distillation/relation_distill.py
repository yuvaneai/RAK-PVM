
import datetime
import os
from argparse import ArgumentParser
import torch.nn.functional as F

import torch

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import  LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.data_module import DataModule
from dataset.pretrain_dataset import CXRMultiLabelPretrainingDataset
from dataset.transforms import DataTransforms

from backbones.text_encoder import BertEncoder

from backbones.anatomy_patch_relevant import  DensePromptForTest

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class SAM_PROMPT(LightningModule):
    def __init__(self,
                 freeze_bert: bool = False,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        self.dense_prompt_test = DensePromptForTest(output_dim=self.hparams.trans_emb_dim)

        pretrain_ckpt = os.path.join(BASE_DIR, f"./result/pretrain/ckpts/example.ckpt")
        our_state_dict = torch.load(
            pretrain_ckpt,
            map_location=torch.device("cpu"))

        text_encoder_q_state_dict = {k.replace("text_encoder_q.", ""): v  for k, v in our_state_dict.items() if k.startswith("text_encoder_q.")}

        self.text_encoder_q.load_state_dict(text_encoder_q_state_dict, strict=False)

        for param in  self.text_encoder_q.parameters():
            param.requires_grad = False


    def forward(self, batch, batch_idx, split="train"):
        image = batch["image"]
        background = batch["background"]

        background_token = self.text_encoder_q.tokenizer(
            background,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=32
        ).to(image.device)
        background_report_feat, background_word_feat, _, _ = self.text_encoder_q(ids=background_token['input_ids'], attn_mask=background_token['attention_mask'],
                                               token_type=background_token['token_type_ids'])

        background_report_feat = F.normalize(background_report_feat, dim=-1)

        B, HW, C = background_word_feat.shape
        dense_for_test = self.dense_prompt_test(batch_size=B)
        dense_for_test = F.normalize(dense_for_test, dim=-1)

        loss_dense_distill = F.mse_loss(dense_for_test, background_report_feat.detach())

        return loss_dense_distill


    def training_step(self, batch, batch_idx):
        loss_dense_distill = self(batch, batch_idx, "train")
        loss = loss_dense_distill
        self.log_dict({"train_loss": loss},
                      sync_dist=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_dense_distill = self(batch, batch_idx, "valid")
        loss =  loss_dense_distill
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
                            default=768, help="64, 128, 256, 512, 768")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.3)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
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
    ckpt_dir = os.path.join(BASE_DIR, f"./result/relation_distill/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger_dir = os.path.join(BASE_DIR, f"./result/relation_distill/log")
    os.makedirs(logger_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name=extension)


    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir, save_top_k=5, save_last=True, mode="min"),
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
