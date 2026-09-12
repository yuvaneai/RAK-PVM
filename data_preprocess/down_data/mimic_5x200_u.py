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


MIMIC_notext_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_u/train_notext.csv"
MIMIC_notext_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_u/test_notext.csv"

MIMICCXR_ORIGINAL_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_u/raw_train.csv"
MIMICCXR_ORIGINAL_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_u/raw_test.csv"

MIMIC_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_u/train_u.csv"
MIMIC_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_u/test_u.csv"

MASTER_TEXT_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/mimic_5*200_u/master_select.csv"


###################################一、选择5个标签的2000个样本作为训练集、200个样本作为测试集，只引入标签信息#################################
# 定义标签映射
label_mapping = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Pleural Effusion": 4
}

def process_data(csv_file, sample_size=200):
    # 读取 CSV 文件
    raw_df = pd.read_csv(csv_file)
    raw_df = raw_df.fillna(0)

    # 需要保留的列
    selected_columns = [
        "Path",
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    # 只保留指定的列
    raw_df_filtered = raw_df[selected_columns]

    # 去掉包含 -1 的行
    raw_df_filtered = raw_df_filtered[~raw_df_filtered[selected_columns[2:]].isin([-1]).any(axis=1)]
    raw_df_filtered = raw_df_filtered[~raw_df_filtered[selected_columns[1:]].isin([-1]).any(axis=1)]
    # 计算每行 1 的数量
    raw_df_filtered["num_ones"] = raw_df_filtered[selected_columns[1:]].sum(axis=1)

    # 只保留 "num_ones == 1" 的行
    raw_df_filtered = raw_df_filtered[raw_df_filtered["num_ones"] == 1].drop(columns=["num_ones"])

    # 转换疾病列为一列
    melted_df = raw_df_filtered.melt(id_vars=["Path"], var_name="Disease", value_name="Label")

    # 只保留 Label == 1 的行
    melted_df = melted_df[melted_df["Label"] == 1].drop(columns=["Label"])

    # 添加 label 列
    melted_df['label'] = melted_df['Disease'].map(label_mapping)

    return melted_df

# 处理数据
train_all_data_df = process_data(MIMICCXR_ORIGINAL_TRAIN_CSV)
test_all_data_df = process_data(MIMICCXR_ORIGINAL_TEST_CSV)

# 随机抽取训练集样本
train_df = train_all_data_df.groupby('Disease').apply(lambda x: x.sample(n=200, random_state=1) if len(x) >= 200 else x).reset_index(drop=True)
# 随机抽取测试集样本
test_df = test_all_data_df.groupby('Disease').apply(lambda x: x.sample(n=20, random_state=1) if len(x) >= 20 else x).reset_index(drop=True)
print(train_df["label"].value_counts())
print(test_df["label"].value_counts())
# 打散样本
train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=1).reset_index(drop=True)

# # 删除 Path 列中的指定前缀
# prefix_to_remove = '/root/data_1/by_data/1_mimic/mimic-cxr-jpg/'
# train_df['Path'] = train_df['Path'].str.replace(prefix_to_remove, '', regex=False)
# test_df['Path'] = test_df['Path'].str.replace(prefix_to_remove, '', regex=False)

print(f"Number of train samples: {len(train_df)}")
print(train_df["label"].value_counts())
print(f"Number of test samples: {len(test_df)}")
print(test_df["label"].value_counts())

train_df.to_csv(MIMIC_notext_TRAIN_CSV)
test_df.to_csv(MIMIC_notext_TEST_CSV)

print(f"Number of train samples: {len(train_df)}")
print(f"Number of valid samples: {len(test_df)}")

###################################二、选择5个标签的2000个样本作为训练集、200个样本作为测试集，引入文本报告信息#################################

notext_train_df = pd.read_csv(MIMIC_notext_TRAIN_CSV)
notext_test_df = pd.read_csv(MIMIC_notext_TEST_CSV)

master_df = pd.read_csv(MASTER_TEXT_CSV)

train_merged_df = pd.merge(notext_train_df, master_df, on="Path", how="inner")
test_merged_df = pd.merge(notext_test_df, master_df, on="Path", how="inner")

train_df = train_merged_df.drop(columns=['subject_id', 'dicom_id', 'study_id'])
test_df = test_merged_df.drop(columns=['subject_id', 'dicom_id', 'study_id'])

# 删除 Path 列中的指定前缀
prefix_to_remove = '/root/data_1/by_data/1_mimic/mimic-cxr-jpg/'
train_df['Path'] = train_df['Path'].str.replace(prefix_to_remove, '', regex=False)
test_df['Path'] = test_df['Path'].str.replace(prefix_to_remove, '', regex=False)

train_df['report'] = train_df['impression'].fillna('') + ' ' + train_df['findings'].fillna('')
test_df['report'] = test_df['impression'].fillna('') + ' ' + test_df['findings'].fillna('')

train_df.to_csv(MIMIC_TRAIN_CSV)
test_df.to_csv(MIMIC_TEST_CSV)

print()

