import pandas as pd
import os
import sys


sys.path.append(os.getcwd())


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MIMICCXR_ORIGINAL_TRAIN_CSV = os.path.join(BASE_DIR, "../pretrain_data/result/2_train_select.csv")
MIMICCXR_ORIGINAL_TEST_CSV = os.path.join(BASE_DIR, "../pretrain_data/result/2_test_select.csv")

MIMIC_TRAIN_CSV = os.path.join(BASE_DIR, "./data_split/mimic_5_200_u/train_u.csv")
MIMIC_TEST_CSV = os.path.join(BASE_DIR, "./data_split/mimic_5_200_u/test_u.csv")

MASTER_TEXT_CSV = os.path.join(BASE_DIR, "../pretrain_data/result/2_master_select.csv")

prefix_to_remove = os.path.join(BASE_DIR, "../pretrain_data/MIMIC_data/mimic-cxr-jpg")

label_mapping = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Pleural Effusion": 4
}

def process_data(csv_file, sample_size=200):

    raw_df = pd.read_csv(csv_file)
    raw_df = raw_df.fillna(0)

    selected_columns = [
        "Path",
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]

    raw_df_filtered = raw_df[selected_columns]

    raw_df_filtered = raw_df_filtered[~raw_df_filtered[selected_columns[2:]].isin([-1]).any(axis=1)]
    raw_df_filtered = raw_df_filtered[~raw_df_filtered[selected_columns[1:]].isin([-1]).any(axis=1)]

    raw_df_filtered["num_ones"] = raw_df_filtered[selected_columns[1:]].sum(axis=1)

    raw_df_filtered = raw_df_filtered[raw_df_filtered["num_ones"] == 1].drop(columns=["num_ones"])

    melted_df = raw_df_filtered.melt(id_vars=["Path"], var_name="Disease", value_name="Label")

    melted_df = melted_df[melted_df["Label"] == 1].drop(columns=["Label"])

    melted_df['label'] = melted_df['Disease'].map(label_mapping)

    return melted_df


train_all_data_df = process_data(MIMICCXR_ORIGINAL_TRAIN_CSV)
test_all_data_df = process_data(MIMICCXR_ORIGINAL_TEST_CSV)

train_df = train_all_data_df.groupby('Disease').apply(lambda x: x.sample(n=200, random_state=1) if len(x) >= 200 else x).reset_index(drop=True)

test_df = test_all_data_df.groupby('Disease').apply(lambda x: x.sample(n=20, random_state=1) if len(x) >= 20 else x).reset_index(drop=True)

print(train_df["label"].value_counts())
print(test_df["label"].value_counts())

train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=1).reset_index(drop=True)

print(f"Number of train samples: {len(train_df)}")
print(train_df["label"].value_counts())
print(f"Number of test samples: {len(test_df)}")
print(test_df["label"].value_counts())

master_df = pd.read_csv(MASTER_TEXT_CSV)

train_merged_df = pd.merge(train_df, master_df, on="Path", how="inner")
test_merged_df = pd.merge(test_df, master_df, on="Path", how="inner")

train_df = train_merged_df.drop(columns=['subject_id', 'dicom_id', 'study_id'])
test_df = test_merged_df.drop(columns=['subject_id', 'dicom_id', 'study_id'])

train_df['Path'] = train_df['Path'].str.replace(prefix_to_remove, '', regex=False)
test_df['Path'] = test_df['Path'].str.replace(prefix_to_remove, '', regex=False)

train_df['report'] = train_df['impression'].fillna('') + ' ' + train_df['findings'].fillna('')
test_df['report'] = test_df['impression'].fillna('') + ' ' + test_df['findings'].fillna('')

train_df.to_csv(MIMIC_TRAIN_CSV)
test_df.to_csv(MIMIC_TEST_CSV)

print()

