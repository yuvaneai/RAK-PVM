from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pathlib import Path
import ast

class UnseenChestx10_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split="train", data_pct=0.01, imsize=224):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.imsize = imsize
        if data_pct != 1 and self.split == "train":
            self.data_df = self.data_df.sample(frac=data_pct, random_state=42)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data_df.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')

        target = torch.tensor(self.data_df.iloc[idx, 4], dtype=torch.long)

        if self.transform:
            image = self.transform(image)


        return image, target

class SeenChestx10_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split="train", data_pct=0.01, imsize=224):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.imsize = imsize
        if data_pct != 1 and self.split == "train":
            self.data_df = self.data_df.sample(frac=data_pct, random_state=42)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.data_df.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')

        target = torch.tensor(self.data_df.iloc[idx, 4], dtype=torch.long)

        if self.transform:
            image = self.transform(image)


        return image, target

class MIMIC5_200_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split="train", data_pct=1.0, imsize=224, max_words=112, sent_num=3) -> None:

        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.imsize = imsize
        if data_pct != 1 and self.split == "train":
            self.data_df = self.data_df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.data_df)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.root_dir, self.data_df.iloc[idx, 2])
        image = Image.open(img_path).convert('RGB')
        # label = self.data_df.iloc[idx, 1]
        target = self.data_df.iloc[idx, 4]
        # report = self.data_df.iloc[idx, 9]
        # ---------- transform ----------
        if self.transform:
            image = self.transform(image)

        return image, target

class SeenChestxray14_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split="train", data_pct=0.01, imsize=224) -> None:

        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.imsize = imsize
        if data_pct != 1 and self.split == "train":
            self.data_df = self.data_df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.data_df.iloc[idx, 2])
        image = Image.open(img_path).convert('RGB')
        # label = self.data_df.iloc[idx, 1]
        target = self.data_df.iloc[idx, 5]
        if self.transform:
            image = self.transform(image)
        return image, target

class UnSeenChestxray14_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split="train", data_pct=0.01, imsize=224) -> None:

        self.data_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.imsize = imsize
        if data_pct != 1 and self.split == "train":
            self.data_df = self.data_df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.data_df.iloc[idx, 2])
        image = Image.open(img_path).convert('RGB')
        # label = self.data_df.iloc[idx, 1]
        target = self.data_df.iloc[idx, 5]
        if self.transform:
            image = self.transform(image)
        return image, target