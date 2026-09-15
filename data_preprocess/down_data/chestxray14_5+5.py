import pandas as pd
from pathlib import Path
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Chestxray_Image_Path = os.path.join(BASE_DIR, "./data/Chestxray/")
Image_Paths_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/chestxray_image_paths.csv")
RAW_Label_CSV = os.path.join(BASE_DIR, "./data/Chestxray/Data_Entry_2017_v2020.csv")
Chestxray_Label_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/chestxray_label.csv")
Chestxray_Label_filter_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/chestxray_label_filter.csv")

Chestxray_RAW_TRAIN_TXT = os.path.join(BASE_DIR, "./data/Chestxray/train_val_list.txt")
Chestxray_RAW_TEST_TXT = os.path.join(BASE_DIR, "./data/Chestxray/test_list.txt")
Chestxray_RAW_TRAIN_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/train_val_list.csv")
Chestxray_RAW_TEST_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/test_list.csv")

Chestxray_SEEN_TRAIN_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/seen_train.csv")
Chestxray_SEEN_TEST_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/seen_test.csv")
Chestxray_UNSEEN_TRAIN_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/unseen_train.csv")
Chestxray_UNSEEN_TEST_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/unseen_test.csv")

Chestxray_SEEN_TRAIN_Proportion_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/seen_train_p.csv")
Chestxray_SEEN_TEST_Proportion_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/seen_test_p.csv")
Chestxray_UNSEEN_TRAIN_Proportion_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/unseen_train_p.csv")
Chestxray_UNSEEN_TEST_Proportion_CSV = os.path.join(BASE_DIR, "./data_split/chestxray14_5+5/unseen_test_p.csv")

###############################1####################################
base_dir = Path(Chestxray_Image_Path)

image_paths = []
for ext in ["*.png", "*.jpg", "*.jpeg"]:
    image_paths.extend(list(base_dir.rglob(ext)))

data = []
for img_path in image_paths:
    data.append({
        "Image Index": img_path.name,
        "Image Path": str(img_path.resolve()),
    })

df = pd.DataFrame(data)
df.to_csv(Image_Paths_CSV, index=False)

##############################2#####################################

image_paths_df = pd.read_csv(Image_Paths_CSV)
raw_label_df = pd.read_csv(RAW_Label_CSV)

merged_df = pd.merge(image_paths_df, raw_label_df, on="Image Index", how="right")

selected_columns = ['Image Index', 'Image Path', 'Finding Labels']
merged_df = merged_df[selected_columns]

merged_df.to_csv(Chestxray_Label_CSV, index=False)

##############################3#####################################

df = pd.read_csv(Chestxray_Label_CSV)

df = df[~df['Finding Labels'].str.contains('\|')]
df = df[df['Finding Labels'] != 'No Finding']

unique_labels = sorted(df['Finding Labels'].unique())
label2id = {label: idx for idx, label in enumerate(unique_labels)}

df['Label_ID'] = df['Finding Labels'].map(label2id)

df.to_csv(Chestxray_Label_filter_CSV, index=False)

##############################4#####################################

with open(Chestxray_RAW_TRAIN_TXT, 'r') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

train_df = pd.DataFrame(lines, columns=["Image Index"])
train_df["split"] = "train"
train_df.to_csv(Chestxray_RAW_TRAIN_CSV, index=False)

with open(Chestxray_RAW_TEST_TXT, 'r') as f:
    lines1 = [line.strip() for line in f.readlines() if line.strip()]

test_df = pd.DataFrame(lines1, columns=["Image Index"])
test_df["split"] = "test"
test_df.to_csv(Chestxray_RAW_TEST_CSV, index=False)

##############################5#####################################

raw_train_df = pd.read_csv(Chestxray_RAW_TRAIN_CSV)
raw_test_df = pd.read_csv(Chestxray_RAW_TEST_CSV)
chestxray_label_df = pd.read_csv(Chestxray_Label_filter_CSV)

train_merged_df = pd.merge(chestxray_label_df, raw_train_df, on="Image Index", how="inner")
test_merged_df = pd.merge(chestxray_label_df, raw_test_df, on="Image Index", how="inner")

seen_labels = {
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Pneumothorax',
    'Effusion',
}

unseen_labels = {
    'Emphysema',
    'Fibrosis',
    'Mass',
    'Nodule',
    'Infiltration',
}

seen_disease_train_df = train_merged_df[train_merged_df['Finding Labels'].isin(seen_labels)]
seen_disease_test_df = test_merged_df[test_merged_df['Finding Labels'].isin(seen_labels)]
unseen_disease_train_df = train_merged_df[train_merged_df['Finding Labels'].isin(unseen_labels)]
unseen_disease_test_df = test_merged_df[test_merged_df['Finding Labels'].isin(unseen_labels)]

seen_label_to_index = {label: index for index, label in enumerate(sorted(seen_labels))}
unseen_label_to_index = {label: index for index, label in enumerate(sorted(unseen_labels))}

seen_disease_train_df = seen_disease_train_df.copy()
seen_disease_test_df = seen_disease_test_df.copy()
unseen_disease_train_df = unseen_disease_train_df.copy()
unseen_disease_test_df = unseen_disease_test_df.copy()

seen_disease_train_df['target'] = seen_disease_train_df['Finding Labels'].map(seen_label_to_index)
seen_disease_train_df = seen_disease_train_df.drop(columns=['Label_ID'])
seen_disease_test_df['target'] = seen_disease_test_df['Finding Labels'].map(seen_label_to_index)
seen_disease_test_df = seen_disease_test_df.drop(columns=['Label_ID'])
unseen_disease_train_df['target'] = unseen_disease_train_df['Finding Labels'].map(unseen_label_to_index)
unseen_disease_train_df = unseen_disease_train_df.drop(columns=['Label_ID'])
unseen_disease_test_df['target'] = unseen_disease_test_df['Finding Labels'].map(unseen_label_to_index)
unseen_disease_test_df = unseen_disease_test_df.drop(columns=['Label_ID'])

print(f"Number of seen_train samples: {len(seen_disease_train_df)}")
print(seen_disease_train_df["target"].value_counts())
print(f"Number of unseen_train samples: {len(unseen_disease_train_df)}")
print(unseen_disease_train_df["target"].value_counts())
print(f"Number of seen_test samples: {len(seen_disease_test_df)}")
print(seen_disease_test_df["target"].value_counts())
print(f"Number of unseen_test samples: {len(unseen_disease_test_df)}")
print(unseen_disease_test_df["target"].value_counts())

seen_disease_train_df.to_csv(Chestxray_SEEN_TRAIN_CSV, index=False)
seen_disease_test_df.to_csv(Chestxray_SEEN_TEST_CSV, index=False)

unseen_disease_train_df.to_csv(Chestxray_UNSEEN_TRAIN_CSV, index=False)
unseen_disease_test_df.to_csv(Chestxray_UNSEEN_TEST_CSV, index=False)

##########################6

def proportional_label_sampling(df, label_col='Disease', sample_size=500, min_samples_per_class=1, random_state=42):
    label_counts = df[label_col].value_counts()
    total = label_counts.sum()
    proportions = label_counts / total

    sampled_list = []

    for label, proportion in proportions.items():
        group = df[df[label_col] == label]
        n_samples = max(int(round(proportion * sample_size)), min_samples_per_class)
        sampled = group.sample(n=min(n_samples, len(group)), random_state=random_state)
        sampled_list.append(sampled)

    sampled_df = pd.concat(sampled_list).reset_index(drop=True)
    return sampled_df

seen_disease_train_df = pd.read_csv(Chestxray_SEEN_TRAIN_CSV)
seen_disease_test_df = pd.read_csv(Chestxray_SEEN_TEST_CSV)
unseen_disease_train_df = pd.read_csv(Chestxray_UNSEEN_TRAIN_CSV)
unseen_disease_test_df = pd.read_csv(Chestxray_UNSEEN_TEST_CSV)

seen_disease_proportion_train_df = proportional_label_sampling(seen_disease_train_df, label_col='target', sample_size=1000)

seen_disease_proportion_test_df = (
    seen_disease_test_df.groupby('target', group_keys=False)
    .apply(lambda x: x.sample(n=min(20, len(x)), random_state=1))
    .reset_index(drop=True)
)
unseen_disease_proportion_train_df = proportional_label_sampling(unseen_disease_train_df, label_col='target', sample_size=1000)

unseen_disease_proportion_test_df = (
    unseen_disease_test_df.groupby('target', group_keys=False)
    .apply(lambda x: x.sample(n=min(20, len(x)), random_state=1))
    .reset_index(drop=True)
)

seen_disease_proportion_train_df = seen_disease_proportion_train_df.sample(frac=1, random_state=1).reset_index(drop=True)
seen_disease_proportion_test_df = seen_disease_proportion_test_df.sample(frac=1, random_state=1).reset_index(drop=True)
unseen_disease_proportion_train_df = unseen_disease_proportion_train_df.sample(frac=1, random_state=1).reset_index(drop=True)
unseen_disease_proportion_test_df = unseen_disease_proportion_test_df.sample(frac=1, random_state=1).reset_index(drop=True)

seen_disease_proportion_train_df.to_csv(Chestxray_SEEN_TRAIN_Proportion_CSV)
seen_disease_proportion_test_df.to_csv(Chestxray_SEEN_TEST_Proportion_CSV)
unseen_disease_proportion_train_df.to_csv(Chestxray_UNSEEN_TRAIN_Proportion_CSV)
unseen_disease_proportion_test_df.to_csv(Chestxray_UNSEEN_TEST_Proportion_CSV)

print(f"Number of seen_train samples: {len(seen_disease_proportion_train_df)}")
print(seen_disease_proportion_train_df["target"].value_counts())
print(f"Number of seen_test samples: {len(seen_disease_proportion_test_df)}")
print(seen_disease_proportion_test_df["target"].value_counts())
print(f"Number of unseen_train samples: {len(unseen_disease_proportion_train_df)}")
print(unseen_disease_proportion_train_df["target"].value_counts())
print(f"Number of unseen_test samples: {len(unseen_disease_proportion_test_df)}")
print(unseen_disease_proportion_test_df["target"].value_counts())