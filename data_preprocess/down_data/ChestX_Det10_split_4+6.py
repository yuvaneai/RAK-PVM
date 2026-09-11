import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
从原始训练集中选取2162张（其中数据提取时按0.01、0.1、1抽取），原始训练集中选取3000张。标签比例1:1。
"""

# COVIDX_ORIGINAL_TRAIN_TXT = COVIDX_DATA_DIR / "train.txt"
ChestX_Det10_RAW_TRAIN_TXT = "/home/by/by/by_data/4_ChestX-Det10-Dataset/ChestX-Det10-Dataset-master/train.json"
# COVIDX_ORIGINAL_TEST_TXT = COVIDX_DATA_DIR / "test.txt"
ChestX_Det10_RAW_TEST_TXT = "/home/by/by/by_data/4_ChestX-Det10-Dataset/ChestX-Det10-Dataset-master/test.json"

ChestX_Det10_SEEN_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestx_det10_4+6/seen_train.csv"
ChestX_Det10_SEEN_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestx_det10_4+6/seen_test.csv"

ChestX_Det10_UNSEEN_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestx_det10_4+6/unseen_train.csv"
ChestX_Det10_UNSEEN_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestx_det10_4+6/unseen_test.csv"

np.random.seed(0)




def preprocess_chsetx_det10():

    raw_train_df = pd.read_json(ChestX_Det10_RAW_TRAIN_TXT)
    raw_test_df = pd.read_json(ChestX_Det10_RAW_TEST_TXT)

    single_disease_train_df = raw_train_df[raw_train_df['syms'].apply(lambda x: len(set(x)) == 1 and len(x) > 0)]
    single_disease_train_df['label'] = single_disease_train_df['syms'].apply(lambda x: x[0] if x else None)

    single_disease_test_df = raw_test_df[raw_test_df['syms'].apply(lambda x: len(set(x)) == 1 and len(x) > 0)]
    single_disease_test_df['label'] = single_disease_test_df['syms'].apply(lambda x: x[0] if x else None)

    # seen_labels = {'Atelectasis', 'Consolidation', 'Fracture', 'Pneumothorax'}
    # unseen_labels = {'Effusion','Emphysema', 'Fibrosis', 'Mass', 'Nodule', 'Calcification'}

    seen_labels = {'Atelectasis', 'Consolidation', 'Fracture', 'Effusion', 'Pneumothorax'}
    unseen_labels = {'Emphysema', 'Fibrosis', 'Mass', 'Nodule', 'Calcification'}

    # 筛选出有效标签的行
    seen_disease_train_df = single_disease_train_df[single_disease_train_df['label'].isin(seen_labels)]
    seen_disease_test_df = single_disease_test_df[single_disease_test_df['label'].isin(seen_labels)]
    unseen_disease_train_df = single_disease_train_df[single_disease_train_df['label'].isin(unseen_labels)]
    unseen_disease_test_df = single_disease_test_df[single_disease_test_df['label'].isin(unseen_labels)]
    # 创建标签到数字的映射字典
    seen_label_to_index = {label: index for index, label in enumerate(sorted(seen_labels))}
    unseen_label_to_index = {label: index for index, label in enumerate(sorted(unseen_labels))}

    # 映射标签到数字
    seen_disease_train_df['target'] = seen_disease_train_df['label'].map(seen_label_to_index)
    seen_disease_test_df['target'] = seen_disease_test_df['label'].map(seen_label_to_index)
    unseen_disease_train_df['target'] = unseen_disease_train_df['label'].map(unseen_label_to_index)
    unseen_disease_test_df['target'] = unseen_disease_test_df['label'].map(unseen_label_to_index)

    print(f"Number of seen_train samples: {len(seen_disease_train_df)}")
    print(seen_disease_train_df["target"].value_counts())
    print(f"Number of unseen_train samples: {len(unseen_disease_train_df)}")
    print(unseen_disease_train_df["target"].value_counts())
    print(f"Number of seen_test samples: {len(seen_disease_test_df)}")
    print(seen_disease_test_df["target"].value_counts())
    print(f"Number of unseen_test samples: {len(unseen_disease_test_df)}")
    print(unseen_disease_test_df["target"].value_counts())


    seen_disease_train_df.to_csv(ChestX_Det10_SEEN_TRAIN_CSV, index=False)
    seen_disease_test_df.to_csv(ChestX_Det10_SEEN_TEST_CSV, index=False)

    unseen_disease_train_df.to_csv(ChestX_Det10_UNSEEN_TRAIN_CSV, index=False)
    unseen_disease_test_df.to_csv(ChestX_Det10_UNSEEN_TEST_CSV, index=False)

    print(seen_disease_train_df.shape)
    print(seen_disease_test_df.shape)

    print(unseen_disease_train_df.shape)
    print(unseen_disease_test_df.shape)


if __name__ == "__main__":
    preprocess_chsetx_det10()