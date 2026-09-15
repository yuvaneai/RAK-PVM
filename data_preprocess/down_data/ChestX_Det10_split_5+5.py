import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ChestX_Det10_RAW_TRAIN_TXT = os.path.join(BASE_DIR, "./data/ChestX-Det10-Dataset/ChestX-Det10-Dataset-master/train.json")
ChestX_Det10_RAW_TEST_TXT = os.path.join(BASE_DIR, "./data/ChestX-Det10-Dataset/ChestX-Det10-Dataset-master/test.json")
ChestX_Det10_SEEN_TRAIN_CSV = os.path.join(BASE_DIR, "./data_split/chestx_det10/seen_train.csv")
ChestX_Det10_SEEN_TEST_CSV = os.path.join(BASE_DIR, "./data_split/chestx_det10/seen_test.csv")
ChestX_Det10_UNSEEN_TRAIN_CSV = os.path.join(BASE_DIR, "./data_split/chestx_det10/unseen_train.csv")
ChestX_Det10_UNSEEN_TEST_CSV = os.path.join(BASE_DIR, "./data_split/chestx_det10/unseen_test.csv")

np.random.seed(0)

def preprocess_chsetx_det10():

    raw_train_df = pd.read_json(ChestX_Det10_RAW_TRAIN_TXT)
    raw_test_df = pd.read_json(ChestX_Det10_RAW_TEST_TXT)

    single_disease_train_df = raw_train_df[raw_train_df['syms'].apply(lambda x: len(set(x)) == 1 and len(x) > 0)]
    single_disease_train_df['label'] = single_disease_train_df['syms'].apply(lambda x: x[0] if x else None)

    single_disease_test_df = raw_test_df[raw_test_df['syms'].apply(lambda x: len(set(x)) == 1 and len(x) > 0)]
    single_disease_test_df['label'] = single_disease_test_df['syms'].apply(lambda x: x[0] if x else None)

    seen_labels = {'Atelectasis', 'Consolidation', 'Fracture', 'Effusion', 'Pneumothorax'}
    unseen_labels = {'Emphysema', 'Fibrosis', 'Mass', 'Nodule', 'Calcification'}

    seen_disease_train_df = single_disease_train_df[single_disease_train_df['label'].isin(seen_labels)]
    seen_disease_test_df = single_disease_test_df[single_disease_test_df['label'].isin(seen_labels)]
    unseen_disease_train_df = single_disease_train_df[single_disease_train_df['label'].isin(unseen_labels)]
    unseen_disease_test_df = single_disease_test_df[single_disease_test_df['label'].isin(unseen_labels)]

    seen_label_to_index = {label: index for index, label in enumerate(sorted(seen_labels))}
    unseen_label_to_index = {label: index for index, label in enumerate(sorted(unseen_labels))}

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