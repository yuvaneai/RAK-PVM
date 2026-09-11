import pandas as pd
import numpy as np

###只处理Triplet_Extraction为全空的行

def clean_triplet_csv(input_file, output_file=None):
    """
    去除Triplet_Extraction列为{}或完全为空的行

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出文件路径，默认添加_cleaned后缀
    """
    # 读取数据
    df = pd.read_csv(input_file)
    original_count = len(df)

    # 确保列存在
    if 'Triplet_Extraction' not in df.columns:
        print(f"错误：文件中没有 'Triplet_Extraction' 列")
        print(f"可用列: {list(df.columns)}")
        return df

    # 处理空值和空字符串
    df['Triplet_Extraction'] = df['Triplet_Extraction'].fillna('')

    # 去除字符串两端的空白字符
    df['Triplet_Extraction'] = df['Triplet_Extraction'].astype(str).str.strip()

    # 创建掩码：要删除的行（{}或完全为空）
    mask_to_remove = (
            df['Triplet_Extraction'].isna() |  # NaN
            (df['Triplet_Extraction'] == '') |  # 空字符串
            (df['Triplet_Extraction'] == '{}')  # 空字典
    )

    # 保留非空行
    df_cleaned = df[~mask_to_remove].reset_index(drop=True)
    cleaned_count = len(df_cleaned)

    # 统计信息
    print("=" * 50)
    print(f"文件清理报告: {input_file}")
    print("=" * 50)
    print(f"原始总行数: {original_count}")
    print(f"清理后行数: {cleaned_count}")
    print(f"删除行数: {original_count - cleaned_count}")
    print("\n删除的行的类型分布:")
    print(f"  - 空字典 {{}}: {(df['Triplet_Extraction'] == '{}').sum()} 行")
    print(f"  - 空字符串: {(df['Triplet_Extraction'] == '').sum()} 行")
    print(f"  - NaN值: {df['Triplet_Extraction'].isna().sum()} 行")
    print("=" * 50)

    # 设置输出文件路径
    if output_file is None:
        import os
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_cleaned{ext}"

    # 保存清理后的文件
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n清理后的文件已保存: {output_file}")

    # 可选：保存被删除的行（用于审计）
    deleted_rows = df[mask_to_remove]
    if len(deleted_rows) > 0:
        audit_file = f"{base}_deleted_rows{ext}"
        deleted_rows.to_csv(audit_file, index=False)
        print(f"被删除的行已保存: {audit_file}")

    return df_cleaned

# 使用示例
cleaned_df = clean_triplet_csv("/root/data/by/3_SAM_prompt/mimic_data/triplet/mimic_merged_triplet.csv")