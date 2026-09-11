import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
# PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = "/root/data/by/by_data/2_siim_20250321/siim/train-rle.csv"

PNEUMOTHORAX_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/siimacr/train.csv"
PNEUMOTHORAX_VALID_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/siimacr/valid.csv"
PNEUMOTHORAX_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/siimacr/test.csv"

PNEUMOTHORAX_MAVL_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/MAVL_split/SIIMACR/train.csv"  # MAVL已经分割好的数据，我们从中取一部分
PNEUMOTHORAX_MAVL_VALID_CSV = "/home/by/by/4_SAM_prompt/0_data_split/MAVL_split/SIIMACR/val.csv"
PNEUMOTHORAX_MAVL_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/MAVL_split/SIIMACR/test.csv"

PNEUMOTHORAX_IMG_DIR = "/home/by/by/by_data/2_siim_20250321/siim/dicom-images-train/"



np.random.seed(0)

def proportional_label_sampling(df, label_col='Disease', sample_size=500, min_samples_per_class=1, random_state=42):
    # 计算标签分布比例
    label_counts = df[label_col].value_counts()
    total = label_counts.sum()
    proportions = label_counts / total

    sampled_list = []

    for label, proportion in proportions.items():
        group = df[df[label_col] == label]
        # 计算每类要抽多少个样本（向上取整，避免0样本）
        n_samples = max(int(round(proportion * sample_size)), min_samples_per_class)
        sampled = group.sample(n=min(n_samples, len(group)), random_state=random_state)
        sampled_list.append(sampled)

    sampled_df = pd.concat(sampled_list).reset_index(drop=True)
    return sampled_df

def preprocess_pneumothorax_data(test_fac=0.15):
    try:
        # df = pd.read_csv(PNEUMOTHORAX_ORIGINAL_TRAIN_CSV)
        train_df = pd.read_csv(PNEUMOTHORAX_MAVL_TRAIN_CSV)
        val_df = pd.read_csv(PNEUMOTHORAX_MAVL_VALID_CSV)
        test_df = pd.read_csv(PNEUMOTHORAX_MAVL_TEST_CSV)
    except:
        raise Exception(
            "Please make sure the the SIIM Pneumothorax dataset is \
            stored at {PNEUMOTHORAX_DATA_DIR}"
        )

    # get image paths
    os.listdir(PNEUMOTHORAX_IMG_DIR)
    img_paths = {}
    for subdir, dirs, files in tqdm(os.walk(PNEUMOTHORAX_IMG_DIR)):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                # img_paths[file_id] = os.path.join(subdir[105:], f)
                img_paths[file_id] = os.path.join(subdir, f)

    # no encoded pixels mean healthy
    train_df["Label"] = train_df.apply(
        lambda x: 0.0 if x["EncodedPixels"] == " -1" else 1.0, axis=1
    )
    test_df["Label"] = test_df.apply(
        lambda x: 0.0 if x["EncodedPixels"] == " -1" else 1.0, axis=1
    )
    val_df["Label"] = val_df.apply(
        lambda x: 0.0 if x["EncodedPixels"] == " -1" else 1.0, axis=1
    )

    train_df["Path"] = train_df["ImageId"].apply(lambda x: img_paths[x])
    test_df["Path"] = test_df["ImageId"].apply(lambda x: img_paths[x])
    val_df["Path"] = val_df["ImageId"].apply(lambda x: img_paths[x])

    # 添加 disease 列
    train_df["disease"] = train_df["Label"].apply(lambda x: "normal" if x == 0.0 else "SIIM-ACR Pneumothorax ")
    val_df["disease"] = val_df["Label"].apply(lambda x: "normal" if x == 0.0 else "SIIM-ACR Pneumothorax ")
    test_df["disease"] = test_df["Label"].apply(lambda x: "normal" if x == 0.0 else "SIIM-ACR Pneumothorax ")

    # 新加筛选 不平衡
    # train_df = proportional_label_sampling(train_df, label_col='Label', sample_size=1000)
    # test_df = proportional_label_sampling(test_df, label_col='Label', sample_size=100)
    # val_df = proportional_label_sampling(val_df, label_col='Label', sample_size=100)
    # 新加筛选 平衡
    # 按类别分别采样 1081 个
    train_0 = train_df[train_df["Label"] == 0].sample(n=500, random_state=42)
    train_1 = train_df[train_df["Label"] == 1].sample(n=500, random_state=42)
    train_df = pd.concat([train_0, train_1]).reset_index(drop=True)

    test_0 = test_df[test_df["Label"] == 0].sample(n=50, random_state=42)
    test_1 = test_df[test_df["Label"] == 1].sample(n=50, random_state=42)
    test_df = pd.concat([test_0, test_1]).reset_index(drop=True)

    train_num_empty_paths = train_df["Path"].isna().sum()
    print(f"train_Path 列中空值的数量: {train_num_empty_paths}")
    test_num_empty_paths = test_df["Path"].isna().sum()
    print(f"test_Path 列中空值的数量: {test_num_empty_paths}")
    val_num_empty_paths = val_df["Path"].isna().sum()
    print(f"val_Path 列中空值的数量: {val_num_empty_paths}")

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Label"].value_counts())
    print(f"Number of valid samples: {len(val_df)}")
    print(val_df["Label"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Label"].value_counts())

    # # ✅ 平衡 test_df 中正负样本数量
    # test_pos = test_df[test_df["Label"] == 1.0]
    # test_neg = test_df[test_df["Label"] == 0.0]
    # min_count = min(len(test_pos), len(test_neg))
    #
    # test_pos_bal = resample(test_pos, replace=False, n_samples=min_count, random_state=42)
    # test_neg_bal = resample(test_neg, replace=False, n_samples=min_count, random_state=42)
    #
    # test_df = pd.concat([test_pos_bal, test_neg_bal]).sample(frac=1, random_state=42).reset_index(drop=True)
    #
    # print(f"Number of test samples (after balance): {len(test_df)}")
    # print(test_df["Label"].value_counts())

    train_df.to_csv(PNEUMOTHORAX_TRAIN_CSV)
    val_df.to_csv(PNEUMOTHORAX_VALID_CSV)
    test_df.to_csv(PNEUMOTHORAX_TEST_CSV)


if __name__ == "__main__":
    preprocess_pneumothorax_data()