import os
import pandas as pd

input_dir = "E:/3_CL_coding/mimic_data_20260304/triplet"                     # 拆分文件所在目录
output_file = "E:/3_CL_coding/mimic_data_20260304/triplet/mimic_merged_triplet.csv"         # 合并后的文件

all_files = sorted([
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.endswith(".csv")
])

print("Found files:")
for f in all_files:
    print(" -", f)

dfs = []

for i, file in enumerate(all_files):
    print(f"Reading file {i+1}/{len(all_files)}: {file}")
    df_part = pd.read_csv(file)
    dfs.append(df_part)

print("Merging...")
df_merged = pd.concat(dfs, ignore_index=True)

print("Saving to:", output_file)
df_merged.to_csv(output_file, index=False)

print("✅ Merge completed!")
