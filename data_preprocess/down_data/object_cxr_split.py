
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

np.random.seed(0)

OBJ_ORIGINAL_TRAIN_CSV = "/home/by/by/by_data/7_object-CXR/train.csv"
OBJ_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/object_cxr/train.csv"
OBJ_VALID_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/object_cxr/valid.csv"
OBJ_ORIGINAL_DEV_CSV = "/home/by/by/by_data/7_object-CXR/dev.csv"
OBJ_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/object_cxr/test.csv"


def main():
    ori_train_df = pd.read_csv(OBJ_ORIGINAL_TRAIN_CSV)
    ori_train_df['annotation'] = ori_train_df['annotation'].fillna('0')
    ori_train_df['Target'] = ori_train_df['annotation'].apply(lambda x: 0 if x.strip() == '0' else 1)
    ori_test_df = pd.read_csv(OBJ_ORIGINAL_DEV_CSV)
    ori_test_df['annotation'] = ori_test_df['annotation'].fillna('0')
    ori_test_df['Target'] = ori_test_df['annotation'].apply(lambda x: 0 if x.strip() == '0' else 1)
    ori_test_df["annotation"] = ori_test_df["annotation"].astype(str).str.strip().replace("0", "0 0 0 0 0")

    train_df, val_df = train_test_split(
        ori_train_df, test_size=0.2, random_state=0)
    test_df = ori_test_df

    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # 训练集中抽取1000个样本，测试集中抽取100个样本
    train_0 = train_df[train_df["Target"] == 0].sample(n=500, random_state=42)
    train_1 = train_df[train_df["Target"] == 1].sample(n=500, random_state=42)
    train_df = pd.concat([train_0, train_1]).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_0 = test_df[test_df["Target"] == 0].sample(n=50, random_state=42)
    test_1 = test_df[test_df["Target"] == 1].sample(n=50, random_state=42)
    test_df = pd.concat([test_0, test_1]).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 打印划分后的样本数量
    print(f"Training set size: {len(train_df)}")
    # print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    train_df.to_csv(OBJ_TRAIN_CSV, index=False)
    # val_df.to_csv(OBJ_VALID_CSV, index=False)
    test_df.to_csv(OBJ_TEST_CSV, index=False)



if __name__ == "__main__":
    main()