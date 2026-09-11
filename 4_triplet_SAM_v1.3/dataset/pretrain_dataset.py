import os

import pickle
from tqdm import tqdm
import numpy as np
from nltk.tokenize import RegexpTokenizer
import re
from transformers import BertTokenizer
import json
import ast

import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MIMIC_CXR_DATA_DIR = "/home/by/by/by_data/1_mimic/mimic-cxr-jpg/"
# MIMIC_CXR_MASTER_TRIPLET_CSV = "/home/by/by/4_SAM_prompt/mimic_data/triplet/mimic_merged_triplet_cleaned.csv"
MIMIC_CXR_MASTER_TRIPLET_CSV = "/home/by/by/4_SAM_prompt/mimic_data_20260304/process/1_mimictriplet.csv"
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL = "Path"
MIMIC_CXR_SPLIT_COL = "split"

def pad_triplets(triplet_list, mask_list):
    """
    triplet_list: List[Tensor] or None, each Tensor [Ni, D]
    mask_list:    List[Tensor] or None, each Tensor [Ni]
    Returns:
        padded: [B, Nmax, D]
        masks:  [B, Nmax]
    """
    # 过滤 None
    valid = [t for t in triplet_list if t is not None]
    if len(valid) == 0:
        return None, None

    max_len = max(t.size(0) for t in valid)
    dim = valid[0].size(1)

    padded, masks = [], []

    for t, m in zip(triplet_list, mask_list):
        if t is None:
            padded.append(torch.zeros(max_len, dim))
            masks.append(torch.zeros(max_len, dtype=torch.bool))
        else:
            pad_len = max_len - t.size(0)
            padded.append(torch.cat([t, torch.zeros(pad_len, dim)], dim=0))
            masks.append(torch.cat([m, torch.zeros(pad_len, dtype=torch.bool)], dim=0))

    return torch.stack(padded), torch.stack(masks)

class CXRMultiLabelPretrainingDataset(data.Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0,
                 root_dir="/home/by/by/by_data/1_mimic/mimic-cxr-jpg/",
                 max_words=112):
        super().__init__()
        if not os.path.exists(MIMIC_CXR_DATA_DIR):
            raise RuntimeError(f"{MIMIC_CXR_DATA_DIR} does not exist!")

        self.transform = transform
        self.root_dir = root_dir

        self.df = pd.read_csv(MIMIC_CXR_MASTER_TRIPLET_CSV)
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]

        # load studies and study to text mapping
        self.filenames, self.path2impression = self.load_impression_data(split)
        self.path2position = self.load_position_data(split)
        self.path2background = self.load_background_data(split)

        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            "/home/by/by/4_SAM_prompt/huggingface_model/Bio_ClinicalBERT/")

        self.max_words = max_words

    def load_position_data(self, split):
        # get study to captions mapping
        # TODO: check this
        # filepath = os.path.join(/home/by/by/4_SAM_prompt/4_triplet_SAM_v1.0/output/
        #     BASE_DIR, "../../data/captions.pickle")
        filepath = "/home/by/by/4_SAM_prompt/4_triplet_SAM_v1.0/output/position.pickle"
        if not os.path.isfile(filepath):
            print(
                f"Position file {filepath} does not exit. Creating position...")
            path2position = self.create_path_2_position_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2position, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2position = pickle.load(f)

        # filter studies to use for current split
        # filenames = []
        # for row in self.df.itertuples():
        #     cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
        #     path = getattr(row, MIMIC_CXR_PATH_COL)
        #     if cur_split == split and path in path2position:
        #         filenames.append(path)

        return path2position

    def load_background_data(self, split):
        # get study to captions mapping
        # TODO: check this
        # filepath = os.path.join(/home/by/by/4_SAM_prompt/4_triplet_SAM_v1.0/output/
        #     BASE_DIR, "../../data/captions.pickle")
        filepath = "/home/by/by/4_SAM_prompt/4_triplet_SAM_v1.0/output/background.pickle"
        if not os.path.isfile(filepath):
            print(
                f"Background file {filepath} does not exit. Creating background...")
            path2background = self.create_path_2_background_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2background, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2background = pickle.load(f)

        # filter studies to use for current split
        # filenames = []
        # for row in self.df.itertuples():
        #     cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
        #     path = getattr(row, MIMIC_CXR_PATH_COL)
        #     if cur_split == split and path in path2position:
        #         filenames.append(path)

        return path2background

    def load_impression_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = "/home/by/by/4_SAM_prompt/4_triplet_SAM_v1.0/output/impression.pickle"
        if not os.path.isfile(filepath):
            print(
                f"Impression file {filepath} does not exit. Creating Impression...")
            path2impression = self.create_path_2_impression_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2impression, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2impression = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2impression:
                filenames.append(path)

        return filenames, path2impression

    def create_path_2_position_mapping(self):
        path2position = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick findings
            findings_triplet = row["findings_triplet"]

            # 原来的错误写法（会报 JSONDecodeError）
            # findings_triplet = json.loads(findings_triplet.replace("'", '"'))

            # ✅ 正确写法：专治你的单引号字典字符串，永不报错
            findings_triplet = ast.literal_eval(findings_triplet)
            # ====================================================================

            # 提取所有 position
            positions = []
            for key in findings_triplet:
                pos = findings_triplet[key]["position"]
                # if pos != "none":  # 可选：去掉无意义的 none
                if pos != "none" and pos not in positions:
                    positions.append(pos)

            # patient_positions.append(positions)

            path2position[row[MIMIC_CXR_PATH_COL]] = positions


        return path2position

    def create_path_2_background_mapping(self):
        path2background = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick findings
            findings_triplet = row["findings_triplet"]

            findings_triplet = ast.literal_eval(findings_triplet)

            """
            把三元组字典 转换成 自然语言医学报告句子
            """
            sentences = []

            for key in findings_triplet:
                item = findings_triplet[key]
                entity = item["entity"]
                position = item["position"]
                existence = item["existence"]

                # 跳过 position = none（可选）
                if position == "none":
                    continue

                # 存在状态 → 单词转换
                if existence == "definitely present":
                    state = "present"
                elif existence == "definitely absent":
                    state = "absent"
                elif existence == "uncertain":
                    state = "uncertain"

                # 拼接成自然语言句子（医学报告标准格式）
                if position and position != "none":
                    sentence = f"{entity.capitalize()} is {state} in the {position}"
                    sentences.append(sentence)

            path2background[row[MIMIC_CXR_PATH_COL]] = sentences

            # # 全部连起来，形成一段完整报告
            # full_background = " ".join(sentences)
            #
            # path2background[row[MIMIC_CXR_PATH_COL]] = full_background

        return path2background

    def create_path_2_impression_mapping(self):
        sent_lens, num_sents = [], []
        path2impression = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression

            impression = row["impression"]

            # use space instead of newline
            impression = impression.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            impression = splitter.split(impression)
            impression = [point.split(".") for point in impression]
            impression = [sent for point in impression for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for imp in impression:
                if len(imp) == 0:
                    continue

                imp = imp.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(imp.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2impression[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2impression


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        # 1. 读取原始路径
        key = self.filenames[index]

        # 2. 加载图片
        # 去掉前缀，只保留相对路径
        relative_path = key.split('mimic-cxr-jpg/')[-1]
        img_path = os.path.join(self.root_dir, relative_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 3. 读取位置信息
        position = self.path2position[key]

        # 4. 读取背景信息
        background = self.path2background[key]
        # separate different sentences
        background = list(filter(lambda x: x != "", background))
        background = " ".join(background)

        # 5. 读取impression信息
        series_impression = self.path2impression[key]
        # separate different sentences
        series_impression = list(filter(lambda x: x != "", series_impression))
        impression = " ".join(series_impression)

        # 6. 返回 (图像, 标签)
        return {
        "image": image,
        "position": position,
        "background": background,
        "impression": impression
    }


def cxr_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    positions = [b["position"] for b in batch]
    backgrounds = [b["background"] for b in batch]
    impressions = [b["impression"] for b in batch]
    return {
    "image": images,
    "position": positions,
    "background": backgrounds,
    "impression": impressions
    }


if __name__ == "__main__":
    from dataset.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset =CXRMultiLabelPretrainingDataset(split="train", transform=transform)
    data = dataset[12]
    print(data)
