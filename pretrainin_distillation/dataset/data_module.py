import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .pretrain_dataset import cxr_collate_fn


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, transforms, data_pct, batch_size, num_workers, crop_size=224):
        super().__init__()

        self.dataset = dataset
        self.transforms = transforms
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size

    def train_dataloader(self):
        if self.transforms:
            transform = self.transforms(True, self.crop_size)
        else:
            transform = None
        
        dataset = self.dataset(
            split="train", transform=transform, data_pct=self.data_pct)

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn = cxr_collate_fn
        )

    def val_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="valid", transform=transform, data_pct=self.data_pct)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn = cxr_collate_fn
        )

    def test_dataloader(self):
        if self.transforms:
            transform = self.transforms(False, self.crop_size)
        else:
            transform = None
        dataset = self.dataset(
            split="test", transform=transform, data_pct=self.data_pct)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=cxr_collate_fn
        )

