import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel

# 示例数据: 多 triplet
triplets = {
    '1': {'entity': 'acute process', 'position': 'cardiopulmonary', 'existence': 'definitely absent'},
    '2': {'entity': 'mild', 'position': 'none', 'existence': 'definitely present'},
    '3': {'entity': 'stable cardiomegaly', 'position': 'none', 'existence': 'definitely present'},
    '4': {'entity': 'increasing fullness', 'position': 'hilar', 'existence': 'definitely present'}
}

# 1️⃣ 收集所有唯一节点
entity_nodes = list({t['entity'] for t in triplets.values()})
position_nodes = list({t['position'] for t in triplets.values()})
existence_nodes = list({t['existence'] for t in triplets.values()})

# 保存索引映射
entity2idx = {e: i for i, e in enumerate(entity_nodes)}
position2idx = {p: i for i, p in enumerate(position_nodes)}
existence2idx = {s: i for i, s in enumerate(existence_nodes)}

# 2️⃣ BERT batch 编码
tokenizer = AutoTokenizer.from_pretrained("/home/by/by/4_SAM_prompt/huggingface_model/Bio_ClinicalBERT/")
model = AutoModel.from_pretrained("/home/by/by/4_SAM_prompt/huggingface_model/Bio_ClinicalBERT/")
model.eval()


def batch_encode(text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token embedding


# 编码所有节点
x_entity = batch_encode(entity_nodes)
x_position = batch_encode(position_nodes)
x_existence = batch_encode(existence_nodes)

# 拼接节点特征矩阵
x = torch.vstack([x_entity, x_position, x_existence])

num_entity = len(entity_nodes)
num_position = len(position_nodes)
num_existence = len(existence_nodes)

# 3️⃣ 构建边 (entity → position, entity → existence)
edges_src = []
edges_dst = []

for t in triplets.values():
    e_idx = entity2idx[t['entity']]
    p_idx = position2idx[t['position']] + num_entity
    s_idx = existence2idx[t['existence']] + num_entity + num_position

    edges_src += [e_idx, e_idx]
    edges_dst += [p_idx, s_idx]

# 可选双向边
# edges_src += edges_dst
# edges_dst += edges_src[:len(edges_src) // 2]

edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

# 4️⃣ 构建 PyG 数据对象
data = Data(x=x, edge_index=edge_index)

print("节点数:", data.num_nodes)
print("边数:", data.num_edges)
print("节点特征维度:", data.x.shape)