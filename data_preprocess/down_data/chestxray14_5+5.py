import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

Chestxray_Image_Path = "/home/by/by/by_data/9_Chestxray/"
Image_Paths_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/chestxray_image_paths.csv"
RAW_Label_CSV = "/home/by/by/by_data/9_Chestxray/Data_Entry_2017_v2020.csv"
Chestxray_Label_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/chestxray_label.csv"
Chestxray_Label_filter_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/chestxray_label_filter.csv"

Chestxray_RAW_TRAIN_TXT = "/home/by/by/by_data/9_Chestxray/train_val_list.txt"
Chestxray_RAW_TEST_TXT = "/home/by/by/by_data/9_Chestxray/test_list.txt"
Chestxray_RAW_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/train_val_list.csv"
Chestxray_RAW_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/test_list.csv"

Chestxray_SEEN_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/seen_train.csv"
Chestxray_SEEN_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/seen_test.csv"

Chestxray_UNSEEN_TRAIN_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/unseen_train.csv"
Chestxray_UNSEEN_TEST_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/unseen_test.csv"

Chestxray_SEEN_TRAIN_Proportion_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/seen_train_p.csv"
Chestxray_SEEN_TEST_Proportion_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/seen_test_p.csv"
Chestxray_UNSEEN_TRAIN_Proportion_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/unseen_train_p.csv"
Chestxray_UNSEEN_TEST_Proportion_CSV = "/home/by/by/4_SAM_prompt/0_data_split/result/chestxray14_5+5/unseen_test_p.csv"

###############################1、读取图像数据路径#####################################
# # 配置路径
# base_dir = Path(Chestxray_Image_Path)
#
# # 递归查找所有.png/.jpg文件
# image_paths = []
# for ext in ["*.png", "*.jpg", "*.jpeg"]:
#     image_paths.extend(list(base_dir.rglob(ext)))
#
# # 构建表格数据
# data = []
# for img_path in image_paths:
#     data.append({
#         "Image Index": img_path.name,
#         # "relative_path": str(img_path.relative_to(base_dir)),
#         "Image Path": str(img_path.resolve()),
#         # "parent_folder": img_path.parent.name
#     })
#
# # 生成DataFrame并保存
# df = pd.DataFrame(data)
# df.to_csv(Image_Paths_CSV, index=False)
# print(f"共找到 {len(df)} 张图片，已保存到 {Image_Paths_CSV}")

##############################2、读取图像数据路径#####################################

# # 读取两个 CSV 文件
# image_paths_df = pd.read_csv(Image_Paths_CSV)
# raw_label_df = pd.read_csv(RAW_Label_CSV)
#
# merged_df = pd.merge(image_paths_df, raw_label_df, on="Image Index", how="right")
#
# # 如果只保留特定列（例如保留列 'id' 和 'value'）
# selected_columns = ['Image Index', 'Image Path', 'Finding Labels']  # 替换成你需要的列名
# merged_df = merged_df[selected_columns]
#
# merged_df.to_csv(Chestxray_Label_CSV, index=False)
#
# print(f"共找到 {len(merged_df)} 张图片，已保存到 {Chestxray_Label_CSV}")

##############################3、处理图像数据路径（只选择单标签）#####################################

# # 读取 CSV 文件（假设名为 data.csv）
# df = pd.read_csv(Chestxray_Label_CSV)
#
# # 筛选出不包含 '|' 的行
# df = df[~df['Finding Labels'].str.contains('\|')]
# df = df[df['Finding Labels'] != 'No Finding']
#
# # 获取唯一标签，并建立映射表
# unique_labels = sorted(df['Finding Labels'].unique())  # 可确保顺序一致
# label2id = {label: idx for idx, label in enumerate(unique_labels)}
#
# # 添加编号列
# df['Label_ID'] = df['Finding Labels'].map(label2id)
#
#
# # 保存处理后的数据（可选）
# df.to_csv(Chestxray_Label_filter_CSV, index=False)
#
# print(f"共找到 {len(df)} 张图片，已保存到 {Chestxray_Label_filter_CSV}")


##############################4、将txt文件转化为csv文件#####################################

# 读取txt文件，每行是一个图像索引
with open(Chestxray_RAW_TRAIN_TXT, 'r') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# 创建DataFrame并保存为CSV
train_df = pd.DataFrame(lines, columns=["Image Index"])
train_df["split"] = "train"
train_df.to_csv(Chestxray_RAW_TRAIN_CSV, index=False)

# 读取txt文件，每行是一个图像索引
with open(Chestxray_RAW_TEST_TXT, 'r') as f:
    lines1 = [line.strip() for line in f.readlines() if line.strip()]

# 创建DataFrame并保存为CSV
test_df = pd.DataFrame(lines1, columns=["Image Index"])
test_df["split"] = "test"
test_df.to_csv(Chestxray_RAW_TEST_CSV, index=False)

##############################5、将过滤出来的chestxray14分成训练集和测试集#####################################

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
    # 'Pneumonia',
    # 'Edema',
    'Effusion',
}

unseen_labels = {

    'Emphysema',
    'Fibrosis',
    'Mass',
    'Nodule',
    # 'Hernia',
    'Infiltration',
    # 'Pleural_Thickening'
}

# 筛选出有效标签的行
seen_disease_train_df = train_merged_df[train_merged_df['Finding Labels'].isin(seen_labels)]
seen_disease_test_df = test_merged_df[test_merged_df['Finding Labels'].isin(seen_labels)]
unseen_disease_train_df = train_merged_df[train_merged_df['Finding Labels'].isin(unseen_labels)]
unseen_disease_test_df = test_merged_df[test_merged_df['Finding Labels'].isin(unseen_labels)]

# 创建标签到数字的映射字典
seen_label_to_index = {label: index for index, label in enumerate(sorted(seen_labels))}
unseen_label_to_index = {label: index for index, label in enumerate(sorted(unseen_labels))}

seen_disease_train_df = seen_disease_train_df.copy()
seen_disease_test_df = seen_disease_test_df.copy()
unseen_disease_train_df = unseen_disease_train_df.copy()
unseen_disease_test_df = unseen_disease_test_df.copy()

# 映射标签到数字
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

##########################6.按比例抽取一部分样本，seen_disease 1500+150，unseen_disease 2000+200

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

# 处理数据

seen_disease_train_df = pd.read_csv(Chestxray_SEEN_TRAIN_CSV)
seen_disease_test_df = pd.read_csv(Chestxray_SEEN_TEST_CSV)
unseen_disease_train_df = pd.read_csv(Chestxray_UNSEEN_TRAIN_CSV)
unseen_disease_test_df = pd.read_csv(Chestxray_UNSEEN_TEST_CSV)

seen_disease_proportion_train_df = proportional_label_sampling(seen_disease_train_df, label_col='target', sample_size=1000)
# seen_disease_proportion_test_df = proportional_label_sampling(seen_disease_test_df, label_col='target', sample_size=100)

# seen_disease_proportion_train_df = (
#     seen_disease_train_df.groupby('target', group_keys=False)
#     .apply(lambda x: x.sample(n=min(200, len(x)), random_state=1))
#     .reset_index(drop=True)
# )
seen_disease_proportion_test_df = (
    seen_disease_test_df.groupby('target', group_keys=False)
    .apply(lambda x: x.sample(n=min(20, len(x)), random_state=1))
    .reset_index(drop=True)
)
unseen_disease_proportion_train_df = proportional_label_sampling(unseen_disease_train_df, label_col='target', sample_size=1000)
# unseen_disease_proportion_test_df = proportional_label_sampling(unseen_disease_test_df, label_col='target', sample_size=100)
# unseen_disease_proportion_train_df = (
#     unseen_disease_train_df.groupby('target', group_keys=False)
#     .apply(lambda x: x.sample(n=min(200, len(x)), random_state=1))
#     .reset_index(drop=True)
# )
unseen_disease_proportion_test_df = (
    unseen_disease_test_df.groupby('target', group_keys=False)
    .apply(lambda x: x.sample(n=min(20, len(x)), random_state=1))
    .reset_index(drop=True)
)

# 打散样本
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