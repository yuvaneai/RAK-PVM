import pandas as pd
import numpy as np
import os
import sys
import tqdm
import argparse

"""
从原始训练集中选取10000张（其中数据提取时按0.01、0.1、1抽取）。标签比例1:1:1:1:1。5种疾病。
"""

sys.path.append(os.getcwd())

MIMIC_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_p/train_p.csv"
MIMIC_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_p/test_p.csv"

MIMIC_TRAIN_with_triplet_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_p/train_p_with_triplet.csv"
MIMIC_TEST_with_triplet_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_p/test_p_with_triplet.csv"

MIMIC_merged_triplet_CSV = "/home/by/by/4_SAM_prompt/mimic_data_20260304/triplet/mimic_merged_triplet.csv"

# 读取数据
df_train = pd.read_csv(MIMIC_TRAIN_CSV)
df_test = pd.read_csv(MIMIC_TEST_CSV)
df_triplet = pd.read_csv(MIMIC_merged_triplet_CSV)
prefix_to_remove = '/root/data_1/by_data/1_mimic/mimic-cxr-jpg/'
df_triplet['Path'] = df_triplet['Path'].str.replace(prefix_to_remove, '', regex=False)

# 按 path 左连接
df_train_merge = pd.merge(
    df_train,
    df_triplet,
    on="Path",
    how="left"
)

df_test_merge = pd.merge(
    df_test,
    df_triplet,
    on="Path",
    how="left"
)

# 保存结果
df_train_merge.to_csv(MIMIC_TRAIN_with_triplet_CSV, index=False)
df_test_merge.to_csv(MIMIC_TEST_with_triplet_CSV, index=False)

print("Finished!")
# print("Train samples:", len(df_train))
# print("Merged samples:", len(df_merge))
# print(df_merge.head())

