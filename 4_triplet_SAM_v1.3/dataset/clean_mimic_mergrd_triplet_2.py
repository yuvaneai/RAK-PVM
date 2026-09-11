import pandas as pd
import numpy as np


def clean_triplet_csv(input_file, output_file=None):
    """
    保留Triplet_Extraction列中同时包含以下三种存在状态的行：
    1. 'existence': 'definitely present'
    2. 'existence': 'definitely absent'
    3. 'existence': 'uncertain'

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出文件路径，默认添加_cleaned后缀
    """
    # 读取数据
    df = pd.read_csv(input_file)
    original_count = len(df)

    if 'Triplet_Extraction' not in df.columns:
        print(f"错误：文件中缺少 'Triplet_Extraction' 列")
        print(f"可用列: {list(df.columns)}")
        return df

    # 处理空值
    df['Triplet_Extraction'] = df['Triplet_Extraction'].fillna('').astype(str)

    def contains_all_three_statuses(triplet_str):
        """
        检查字符串中是否同时包含三种存在状态
        """
        if not triplet_str or triplet_str.strip() == '' or triplet_str.strip() == '{}':
            return False

        # 检查三种状态（支持两种引号格式）
        has_present = (
                "'existence': 'definitely present'" in triplet_str or
                '"existence": "definitely present"' in triplet_str
        )

        has_absent = (
                "'existence': 'definitely absent'" in triplet_str or
                '"existence": "definitely absent"' in triplet_str
        )

        has_uncertain = (
                "'existence': 'uncertain'" in triplet_str or
                '"existence": "uncertain"' in triplet_str
        )

        return has_present and has_absent and has_uncertain

    # 创建掩码：同时包含三种状态的行
    mask = df['Triplet_Extraction'].apply(contains_all_three_statuses)

    # 保留同时包含三种状态的行
    df_cleaned = df[mask].reset_index(drop=True)
    cleaned_count = len(df_cleaned)

    # 详细统计信息
    print("=" * 70)
    print(f"文件清理报告: {input_file}")
    print("=" * 70)
    print(f"原始总行数: {original_count}")
    print(f"清理后行数: {cleaned_count}")
    print(f"删除行数: {original_count - cleaned_count}")

    # 统计每种状态的出现情况
    df['has_present'] = df['Triplet_Extraction'].apply(
        lambda x: "'existence': 'definitely present'" in str(x) or '"existence": "definitely present"' in str(x)
    )
    df['has_absent'] = df['Triplet_Extraction'].apply(
        lambda x: "'existence': 'definitely absent'" in str(x) or '"existence": "definitely absent"' in str(x)
    )
    df['has_uncertain'] = df['Triplet_Extraction'].apply(
        lambda x: "'existence': 'uncertain'" in str(x) or '"existence": "uncertain"' in str(x)
    )

    print(f"\n各状态分布情况（清理前）:")
    print(
        f"- 包含 'definitely present' 的行数: {df['has_present'].sum()} ({df['has_present'].sum() / original_count * 100:.1f}%)")
    print(
        f"- 包含 'definitely absent' 的行数: {df['has_absent'].sum()} ({df['has_absent'].sum() / original_count * 100:.1f}%)")
    print(
        f"- 包含 'uncertain' 的行数: {df['has_uncertain'].sum()} ({df['has_uncertain'].sum() / original_count * 100:.1f}%)")

    print(f"\n组合情况（清理前）:")
    print(f"- 同时包含三种状态的行: {mask.sum()} ({mask.sum() / original_count * 100:.2f}%)")
    print(f"- 包含has_present和has_absent两种状态的行: {((df['has_present'] & df['has_absent'])).sum()}")
    print(f"- 仅包含两种状态的行: {((df['has_present'] & df['has_absent'] & ~df['has_uncertain']) |(df['has_present'] & ~df['has_absent'] & df['has_uncertain']) |(~df['has_present'] & df['has_absent'] & df['has_uncertain'])).sum()}")
    print(f"- 仅包含一种状态的行: {((df['has_present'] & ~df['has_absent'] & ~df['has_uncertain']) |(~df['has_present'] & df['has_absent'] & ~df['has_uncertain']) |(~df['has_present'] & ~df['has_absent'] & df['has_uncertain'])).sum()}")

    # 清理前的示例
    print(f"\n清理前的部分示例:")
    print("-" * 40)
    for i in range(min(5, len(df))):
        statuses = []
        if df['has_present'].iloc[i]:
            statuses.append("present")
        if df['has_absent'].iloc[i]:
            statuses.append("absent")
        if df['has_uncertain'].iloc[i]:
            statuses.append("uncertain")

        sample = df['Triplet_Extraction'].iloc[i]
        if len(sample) > 150:
            sample = sample[:150] + "..."

        print(f"行 {i + 1} [包含: {', '.join(statuses) if statuses else '无状态'}]")
        print(f"  内容: {sample}")
        print()

    # 清理后的示例
    if cleaned_count > 0:
        print(f"\n清理后的部分示例（同时包含三种状态）:")
        print("-" * 40)
        for i in range(min(5, cleaned_count)):
            sample = df_cleaned['Triplet_Extraction'].iloc[i]
            if len(sample) > 150:
                sample = sample[:150] + "..."

            print(f"行 {i + 1}: {sample}")
            print()
    else:
        print(f"\n警告：没有找到同时包含三种状态的行！")

    # 删除临时列
    df_cleaned = df_cleaned.drop(['has_present', 'has_absent', 'has_uncertain'], axis=1, errors='ignore')

    print("=" * 70)

    # 设置输出文件路径
    if output_file is None:
        import os
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_all_three_statuses{ext}"

    # 保存清理后的文件
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n清理后的文件已保存: {output_file}")
    print(f"保留了 {cleaned_count} 行同时包含三种存在状态的数据")

    return df_cleaned


# 使用示例
cleaned_df = clean_triplet_csv("/root/data/by/3_SAM_prompt/mimic_data/triplet/mimic_merged_triplet.csv")