# 文件：feature_extract/BERT.py
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os
import sys
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==== 本地 DNABERT 路径，请保持你原有配置 ====
local_model_path = r"D:\my code\models\dna_bert_6"

tokenizer = BertTokenizer.from_pretrained(local_model_path)
bert_model = BertModel.from_pretrained(local_model_path).to(device)
bert_model.eval()

def dna_to_text(seq: str) -> str:
    return " ".join(list(seq))

def get_bert_embeddings(sequences, seq_length=41, output_dim=24):
    """
    将每个 token 的 768 维表示规整到 seq_length，并通过“分组均值”近似降到 output_dim 维；
    再做 StandardScaler 归一化（在当前批上拟合），减少量纲差异。
    """
    all_embeddings = []

    with torch.no_grad():
        for seq in tqdm(sequences, desc="Encoding with DNABERT"):
            text = dna_to_text(seq)
            inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                               max_length=seq_length + 2, truncation=True)  # +2: [CLS],[SEP]
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = bert_model(**inputs)  # last_hidden_state: [1, L+2, 768]
            token_embeddings = outputs.last_hidden_state[:, 1:-1, :]  # [1, L, 768]
            # pad/trunc 到固定 L=seq_length
            if token_embeddings.size(1) < seq_length:
                pad = torch.zeros(1, seq_length - token_embeddings.size(1), token_embeddings.size(2), device=device)
                token_embeddings = torch.cat([token_embeddings, pad], dim=1)
            elif token_embeddings.size(1) > seq_length:
                token_embeddings = token_embeddings[:, :seq_length, :]

            # 768 -> output_dim（分组均值，信息保留度 > 原 8 维）
            group_size = token_embeddings.size(2) // output_dim
            reduced = []
            for i in range(output_dim):
                s = i * group_size
                e = (i + 1) * group_size if i < output_dim - 1 else token_embeddings.size(2)
                reduced.append(token_embeddings[:, :, s:e].mean(dim=2, keepdim=True))  # [1, L, 1]
            reduced_embeddings = torch.cat(reduced, dim=2)  # [1, L, output_dim]

            all_embeddings.append(reduced_embeddings.squeeze(0).cpu().numpy())

    arr = np.array(all_embeddings)  # [N, L, output_dim]
    # 标准化到 0 均值/1 方差（按特征维度展平后拟合）
    N, L, D = arr.shape
    scaler = StandardScaler()
    arr_2d = arr.reshape(N * L, D)
    arr_2d = scaler.fit_transform(arr_2d)
    arr = arr_2d.reshape(N, L, D)
    return arr

def Bert_out(dataset_name: str, seq_length=41, output_dim=24):
    """返回 (train_embeddings, test_embeddings)，形状分别为 [N_train, L, D] 与 [N_test, L, D]"""
    if dataset_name == "Dataset_mouse":
        train_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_positive.npy')
        train_seq_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_negative.npy')
        test_seq_positive_path  = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_seq_positive.npy')
        test_seq_negative_path  = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_seq_negative.npy')
    else:
        raise ValueError("Unknown dataset name.")

    # 加载序列
    train_pos = np.load(train_seq_positive_path).tolist()
    train_neg = np.load(train_seq_negative_path).tolist()
    test_pos  = np.load(test_seq_positive_path).tolist()
    test_neg  = np.load(test_seq_negative_path).tolist()

    train_sequences = np.concatenate([train_pos, train_neg], axis=0)
    test_sequences  = np.concatenate([test_pos,  test_neg],  axis=0)

    train_embeddings = get_bert_embeddings(train_sequences, seq_length=seq_length, output_dim=output_dim)
    test_embeddings  = get_bert_embeddings(test_sequences,  seq_length=seq_length, output_dim=output_dim)

    return train_embeddings, test_embeddings
