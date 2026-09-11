import os
import pandas as pd
import ast

input_dir = "E:/3_CL_coding/mimic_data_20260304/triplet/mimic_merged_triplet.csv"                     # 拆分文件所在目录
output_file = "E:/3_CL_coding/mimic_data_20260304/process/1_mimictriplet.csv"         # 合并后的文件

df = pd.read_csv(input_dir)

# def not_empty(x):
#     try:
#         return len(ast.literal_eval(x)) > 0
#     except:
#         return False
#
# df = df[df['impression_triplet'].apply(not_empty) |
#         df['findings_triplet'].apply(not_empty)]

df = df[(df['impression_triplet'] != '{}') & (df['findings_triplet'] != '{}')]

df.to_csv(output_file, index=False)
