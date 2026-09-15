import os
from torchvision import datasets
from torch.utils.data import DataLoader
from .augmentation import train_augmentation, test_augmentation
from .dataset import (
                      UnseenChestx10_Dataset, SeenChestx10_Dataset,  MIMIC5_200_Dataset, SeenChestxray14_Dataset, UnSeenChestxray14_Dataset
                      )
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_unseenchestx10(datadir: str, data_pct, img_size: int, mean: tuple, std: tuple):
    trainset = UnseenChestx10_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/chestx_det10/unseen_train.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data/ChestX-Det10-Dataset/train-old"),
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        split="train",
        data_pct=data_pct
    )
    testset = UnseenChestx10_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/chestx_det10/unseen_test.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data/ChestX-Det10-Dataset/test_data"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        split="test"
    )
    return trainset, testset

def load_seenchestx10(datadir: str, data_pct, img_size: int, mean: tuple, std: tuple):
    trainset = SeenChestx10_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/chestx_det10/seen_train.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data/ChestX-Det10-Dataset/train-old"),
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        split="train",
        data_pct=data_pct
    )
    testset = SeenChestx10_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/chestx_det10/seen_test.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data/ChestX-Det10-Dataset/test_data"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        split="test"
    )
    return trainset, testset

def load_mimic5_200_p(datadir: str, data_pct, img_size: int, mean: tuple, std: tuple):
    trainset = MIMIC5_200_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/mimic_5_200_p/train_p.csv"),  # proportion
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/pretrain_data/MIMIC_data/mimic-cxr-jpg"),
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        split="train",
        data_pct=data_pct
    )

    testset = MIMIC5_200_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/mimic_5_200_p/test_p.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/pretrain_data/MIMIC_data/mimic-cxr-jpg"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        split="test"
    )
    return trainset, testset

def load_mimic5_200_u(datadir: str, data_pct, img_size: int, mean: tuple, std: tuple):
    trainset = MIMIC5_200_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/mimic_5_200_u/train_u.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/pretrain_data/MIMIC_data/mimic-cxr-jpg"),
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        split="train",
        data_pct=data_pct
    )

    testset = MIMIC5_200_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/mimic_5_200_u/test_u.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/pretrain_data/MIMIC_data/mimic-cxr-jpg"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        split="test"
    )
    return trainset, testset

def load_seenchestxray14(datadir: str, data_pct, img_size: int, mean: tuple, std: tuple):
    trainset = SeenChestxray14_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/chestxray14_5+5/seen_train_p.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data/Chestxray"),
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        split="train",
        data_pct=data_pct
    )
    testset = SeenChestxray14_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/chestxray14_5+5/seen_test_p.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data/Chestxray"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        split="test"
    )
    return trainset, testset

def load_unseenchestxray14(datadir: str, data_pct, img_size: int, mean: tuple, std: tuple):
    trainset = UnSeenChestxray14_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/chestxray14_5+5/unseen_train_p.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data/Chestxray"),
        transform=train_augmentation(img_size=img_size, mean=mean, std=std),
        split="train",
        data_pct=data_pct
    )
    testset = UnSeenChestxray14_Dataset(
        csv_file=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data_split/chestxray14_5+5/unseen_test_p.csv"),
        root_dir=os.path.join(BASE_DIR, "../../data_preprocess/down_data/data/Chestxray"),
        transform=test_augmentation(img_size=img_size, mean=mean, std=std),
        split="test"
    )
    return trainset, testset

def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=16
    )
