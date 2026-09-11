import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
从原始训练集中选取2162张（其中数据提取时按0.01、0.1、1抽取），原始测试集中选取3000张。标签比例1:1。
"""

COVIDX_ORIGINAL_TRAIN_TXT = "/home/by/by/by_data/5_covid_x/train.txt"
COVIDX_ORIGINAL_TEST_TXT = "/home/by/by/by_data/5_covid_x/test.txt"
COVIDX_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/covidx/covidx_train.csv"
COVIDX_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/covidx/covidx_test.csv"



np.random.seed(0)

def set_label(x):
    if x == "positive":
        return 1
    elif x == "negative":
        return 0


def preprocess_covidx():

    raw_train_df = pd.read_csv(COVIDX_ORIGINAL_TRAIN_TXT, sep=" ", header=None)

    raw_train_df.columns = ['patient id', 'filename', 'class', 'data source']

    raw_train_df = raw_train_df.drop(['patient id', 'data source'], axis=1)
    raw_train_df["labels"] = raw_train_df["class"].apply(set_label)

    # 按类别分别采样 1081 个
    train_0 = raw_train_df[raw_train_df["labels"] == 0].sample(n=1081, random_state=42)  # 2162+3000  6469+2317
    train_1 = raw_train_df[raw_train_df["labels"] == 1].sample(n=1081, random_state=42)

    # 合并数据集
    train_df = pd.concat([train_0, train_1]).reset_index(drop=True)
    train_df.to_csv(COVIDX_TRAIN_CSV, index=False)

    # 在剩余数据中抽取测试集
    remaining_df = raw_train_df.drop(train_df.index)  # 移除已选入训练集的数据
    test_0 = remaining_df[remaining_df["labels"] == 0].sample(n=1500, random_state=42)
    test_1 = remaining_df[remaining_df["labels"] == 1].sample(n=1500, random_state=42)
    test_df = pd.concat([test_0, test_1]).reset_index(drop=True)

    # raw_test_df = pd.read_csv(COVIDX_ORIGINAL_TEST_TXT, sep=" ", header=None)
    # raw_test_df.columns = ['patient id', 'filename', 'class', 'data source']
    # raw_test_df = raw_test_df.drop(['patient id', 'data source'], axis=1)
    # raw_test_df["labels"] = raw_test_df["class"].apply(set_label)
    # # 从测试集中均衡抽取
    # test_0 = raw_test_df[raw_test_df["labels"] == 0].sample(n=1500, random_state=42)
    # test_1 = raw_test_df[raw_test_df["labels"] == 1].sample(n=1500, random_state=42)
    # test_df = pd.concat([test_0, test_1]).reset_index(drop=True)
    test_df.to_csv(COVIDX_TEST_CSV, index=False)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["labels"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["labels"].value_counts())
    print(train_df.shape)
    print(test_df.shape)


if __name__ == "__main__":
    preprocess_covidx()