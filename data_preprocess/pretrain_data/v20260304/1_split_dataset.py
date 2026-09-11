import pandas as pd
import math
import os

input_file = "E:/3_CL_coding/mimic_data_20260304/0_raw_merged_report.csv"  # 你的原始数据
output_dir = "E:/3_CL_coding/mimic_data_20260304/splits"
chunk_size = 30000  # 每份 3 万条

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)
num_rows = len(df)
num_chunks = math.ceil(num_rows / chunk_size)

print(f"Total rows: {num_rows}, Splitting into {num_chunks} parts")

for i in range(num_chunks):
    start = i * chunk_size
    end = min(start + chunk_size, num_rows)
    df_chunk = df.iloc[start:end]

    output_path = os.path.join(output_dir, f"raw_mimic_part_{i}.csv")
    df_chunk.to_csv(output_path, index=False)

    print(f"Saved: {output_path} ({start} → {end})")
