import pickle
import numpy as np
import pandas as pd
# from mgca.constants import *
from sklearn.model_selection import train_test_split

"""
从原始训练集中选取8,486张（其中数据提取时按0.01、0.1、1抽取），原始训练集中选取3538张。标签比例1:1。
"""

np.random.seed(0)

RSNA_ORIGINAL_TRAIN_CSV = "/home/by/by/by_data/6_RSNA-2018/stage_2_train_labels.csv"
# 输出
RSNA_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/rsna/train.csv"
RSNA_VALID_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/rsna/val.csv"
RSNA_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/rsna/test.csv"


# create bounding boxes
def create_bbox(row):
    if row["Target"] == 0:
        return 0
    else:
        x1 = row["x"]
        y1 = row["y"]
        x2 = x1 + row["width"]
        y2 = y1 + row["height"]
        return [x1, y1, x2, y2]


def preprocess_rsna_data(random_seed=42):
    """预处理 RSNA 数据，确保训练集和测试集完全不重叠，并固定抽样数量"""
    try:
        df = pd.read_csv(RSNA_ORIGINAL_TRAIN_CSV)
    except FileNotFoundError:
        raise Exception(f"请确保 RSNA 数据集存储在 {RSNA_ORIGINAL_TRAIN_CSV}")

    # 生成 bbox 标签
    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # 按 patientId 进行聚合，处理多个 bbox
    df = df.groupby("patientId")["bbox"].agg(list).reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # 生成目标标签 (0: 无病变, 1: 有病变)
    df["Target"] = df["bbox"].apply(lambda x: 0 if x is None else 1)

    # df['bbox'] = df['bbox'].fillna('0')
    # df['bbox'] = df['bbox'].astype(str).str.strip().replace("0", "[[0.0, 0.0, 0.0, 0.0]]")

    # 确保数据唯一，不重复划分
    df = df.drop_duplicates(subset=["patientId"])

    # 按类别分别采样，确保训练集和测试集不重叠
    train_0 = df[df["Target"] == 0].sample(n=3234, random_state=random_seed)  # 800+200  500+500  6469+2317
    train_1 = df[df["Target"] == 1].sample(n=3235, random_state=random_seed)
    train_df = pd.concat([train_0, train_1]).reset_index(drop=True)

    # 在剩余数据中抽取测试集
    remaining_df = df.drop(train_df.index)  # 移除已选入训练集的数据
    test_0 = remaining_df[remaining_df["Target"] == 0].sample(n=1158, random_state=random_seed) # 1500
    test_1 = remaining_df[remaining_df["Target"] == 1].sample(n=1159, random_state=random_seed)
    test_df = pd.concat([test_0, test_1]).reset_index(drop=True)

    # 添加 disease 列
    train_df["disease"] = train_df["Target"].apply(lambda x: "normal" if x == 0.0 else "RSNA Pneumonia")
    test_df["disease"] = test_df["Target"].apply(lambda x: "normal" if x == 0.0 else "RSNA Pneumonia")

    # 打散样本
    train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=1).reset_index(drop=True)

    # 保存数据
    train_df.to_csv(RSNA_TRAIN_CSV, index=False)
    test_df.to_csv(RSNA_TEST_CSV, index=False)

    # 数据统计
    print(f"✅ 训练集样本数: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"✅ 测试集样本数: {len(test_df)}")
    print(test_df["Target"].value_counts())


if __name__ == "__main__":
    preprocess_rsna_data()
