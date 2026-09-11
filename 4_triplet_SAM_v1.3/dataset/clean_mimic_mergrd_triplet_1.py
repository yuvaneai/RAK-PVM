import pandas as pd
import numpy as np

###处理Triplet_Extraction为全空的行, 还有present为全空的行

import pandas as pd
import numpy as np


def clean_triplet_csv_simple(input_file, output_file=None):
    """
    简单版本：删除Triplet_Extraction列中不包含'existence': 'definitely present'文本的行
    """
    # 读取数据
    df = pd.read_csv(input_file)
    original_count = len(df)

    if 'Triplet_Extraction' not in df.columns:
        print(f"错误：文件中缺少 'Triplet_Extraction' 列")
        return df

    # 处理空值
    df['Triplet_Extraction'] = df['Triplet_Extraction'].fillna('').astype(str)

    # 创建掩码：包含'definitely present'文本的行
    # 检查两种可能的引号格式
    mask = (
            df['Triplet_Extraction'].str.contains("'existence': 'definitely present'", na=False) |
            df['Triplet_Extraction'].str.contains('"existence": "definitely present"', na=False)
    )

    # 保留包含'definitely present'的行
    df_cleaned = df[mask].reset_index(drop=True)
    cleaned_count = len(df_cleaned)

    # 统计信息
    print("=" * 60)
    print(f"文件清理报告: {input_file}")
    print("=" * 60)
    print(f"原始总行数: {original_count}")
    print(f"清理后行数: {cleaned_count}")
    print(f"删除行数: {original_count - cleaned_count}")

    # 设置输出文件路径
    if output_file is None:
        import os
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_with_definitely_present{ext}"

    # 保存清理后的文件
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n清理后的文件已保存: {output_file}")

    return df_cleaned

# 使用示例
cleaned_df = clean_triplet_csv_simple("/root/data/by/3_SAM_prompt/mimic_data/triplet/mimic_merged_triplet.csv")