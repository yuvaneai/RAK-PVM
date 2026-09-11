import pandas as pd
import numpy as np

# ===================== 1. 读取CSV文件 =====================
# 请替换为你的CSV文件路径
input_file = "E:/3_CL_coding/mimic_data_20260304/0_raw_merged_report.csv"
output_file = "E:/3_CL_coding/mimic_data_20260304/1_del_any_none_report.csv"

# 读取CSV
df = pd.read_csv(input_file)

# 查看列名，确认impression和findings对应的列名（关键！）
print("数据集列名：")
print(df.columns.tolist())

# ===================== 2. 数据清洗 - 删除空值行 =====================
# 请根据实际列名调整！以下假设列名分别为'impression'和'findings'
# 如果列名不是这个，请替换为实际列名（比如第7列/第8列）

# 方案1：如果知道列名（推荐）
# 先去除字符串两端的空格
df['impression'] = df['impression'].astype(str).str.strip()
df['findings'] = df['findings'].astype(str).str.strip()

# 方案2：如果不知道列名，根据位置选择（比如impression是第7列，findings是第8列）
# df['impression'] = df.iloc[:, 7].astype(str).str.strip()
# df['findings'] = df.iloc[:, 8].astype(str).str.strip()

# 筛选条件：impression非空 且 findings非空
# 排除空字符串、仅空格、'nan'（字符串型）、NaN等情况
filter_condition = (
    (df['impression'] != '') &
    (df['impression'] != 'nan') &
    (df['findings'] != '') &
    (df['findings'] != 'nan') &
    (df['findings'] != '.')
    & (df['findings'] != '..')
    & (df['findings'] != '___')

)

# 应用筛选条件
df_cleaned = df[filter_condition].copy()

# ===================== 3. 输出处理结果 =====================
print(f"\n原始数据行数：{len(df)}")
print(f"清洗后数据行数：{len(df_cleaned)}")
print(f"删除的空值行数：{len(df) - len(df_cleaned)}")

# ===================== 4. 保存清洗后的数据 =====================
df_cleaned.to_csv(output_file)

print(f"\n清洗后的数据已保存至：{output_file}")