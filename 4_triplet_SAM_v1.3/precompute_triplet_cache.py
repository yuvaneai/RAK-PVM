import os
import ast
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from backbones.text_encoder import BertEncoder

CSV_PATH = "/root/data/by/3_SAM_prompt/mimic_data/triplet/mimic_merged_triplet_cleaned.csv"
SAVE_DIR = "/root/data/by/3_SAM_prompt/mimic_data/triplet_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

output_dim=128
freeze_bert=False  # 冻结bert

text_encoder_q = BertEncoder(
    output_dim=128, freeze_bert=False).to(DEVICE)


def split_triplets(triplet_str):
    if len(triplet_str) == 0:
        print("有全空的值，必须重新处理")

    present, absent, uncertain = [], [], []
    triplet_dict = ast.literal_eval(triplet_str)

    for v in triplet_dict.values():
        text = v["entity"]
        if v["position"] != "none":
            text = f"{text} in {v['position']}"

        if v["existence"] == "definitely present":
            present.append(text)
        elif v["existence"] == "definitely absent":
            absent.append(text)
        elif v["existence"] == "uncertain":
            uncertain.append(text)

    # 正确写法 - 每个都应该独立检查
    if len(present) == 0:
        present.append("None")
    if len(absent) == 0:
        absent.append("None")
    if len(uncertain) == 0:
        uncertain.append("None")

    return present, absent, uncertain


@torch.no_grad()
def encode(text_list):
    if len(text_list) == 0:
        return None, None

    inputs = text_encoder_q.tokenizer(
        text_list,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    report_feat, word_feat, last_atten_pt, sents = text_encoder_q(
        ids=inputs['input_ids'],
        attn_mask=inputs['attention_mask'],
        token_type=inputs['token_type_ids']
    )
    embeds = report_feat  # CLS
    mask = torch.ones(len(text_list), dtype=torch.bool)

    return embeds.cpu(), mask


df = pd.read_csv(CSV_PATH)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    present, absent, uncertain = split_triplets(row["Triplet_Extraction"])

    p_emb, p_mask = encode(present)
    a_emb, a_mask = encode(absent)
    u_emb, u_mask = encode(uncertain)

    torch.save({
        "present_embeds": p_emb,
        "present_mask": p_mask,
        "absent_embeds": a_emb,
        "absent_mask": a_mask,
        "uncertain_embeds": u_emb,
        "uncertain_mask": u_mask,
    }, os.path.join(SAVE_DIR, f"{idx}.pt"))
