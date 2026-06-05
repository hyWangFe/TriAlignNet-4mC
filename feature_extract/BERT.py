import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

local_model_path = r"D:\my code\models\dna_bert_6"

tokenizer = BertTokenizer.from_pretrained(local_model_path)
bert_model = BertModel.from_pretrained(local_model_path).to(device)
bert_model.eval()


def dna_to_text(seq: str) -> str:
    return " ".join(list(seq))


def get_bert_embeddings(sequences, seq_length=41, output_dim=24):
    all_embeddings = []

    with torch.no_grad():
        for seq in tqdm(sequences, desc="Encoding with DNABERT"):
            text = dna_to_text(seq)
            inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                               max_length=seq_length + 2, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = bert_model(**inputs)
            token_embeddings = outputs.last_hidden_state[:, 1:-1, :]

            if token_embeddings.size(1) < seq_length:
                pad = torch.zeros(1, seq_length - token_embeddings.size(1), token_embeddings.size(2), device=device)
                token_embeddings = torch.cat([token_embeddings, pad], dim=1)
            elif token_embeddings.size(1) > seq_length:
                token_embeddings = token_embeddings[:, :seq_length, :]

            group_size = token_embeddings.size(2) // output_dim
            reduced = []
            for i in range(output_dim):
                s = i * group_size
                e = (i + 1) * group_size if i < output_dim - 1 else token_embeddings.size(2)
                reduced.append(token_embeddings[:, :, s:e].mean(dim=2, keepdim=True))
            reduced_embeddings = torch.cat(reduced, dim=2)

            all_embeddings.append(reduced_embeddings.squeeze(0).cpu().numpy())

    # 【修复】：直接返回未归一化的原始数据
    arr = np.array(all_embeddings)
    return arr


def Bert_out(dataset_name: str, seq_length=41, output_dim=24):
    if dataset_name == 'Dataset_mouse':
        base_dir = os.path.join(root_dir, 'data/Dataset_mouse/npy')
    else:
        base_dir = os.path.join(root_dir, f'data/Dataset_6species/npy/{dataset_name}')

    train_seq_positive_path = os.path.join(base_dir, 'train_seq_positive.npy')
    train_seq_negative_path = os.path.join(base_dir, 'train_seq_negative.npy')
    test_seq_positive_path = os.path.join(base_dir, 'test_seq_positive.npy')
    test_seq_negative_path = os.path.join(base_dir, 'test_seq_negative.npy')

    train_pos = np.load(train_seq_positive_path).tolist()
    train_neg = np.load(train_seq_negative_path).tolist()
    test_pos = np.load(test_seq_positive_path).tolist()
    test_neg = np.load(test_seq_negative_path).tolist()

    train_sequences = np.concatenate([train_pos, train_neg], axis=0)
    test_sequences = np.concatenate([test_pos, test_neg], axis=0)

    train_embeddings = get_bert_embeddings(train_sequences, seq_length=seq_length, output_dim=output_dim)
    test_embeddings = get_bert_embeddings(test_sequences, seq_length=seq_length, output_dim=output_dim)

    return train_embeddings, test_embeddings