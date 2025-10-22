# 文件：feature_extract/Bio_feature.py
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import os, sys
from itertools import product
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 基础特征 ---
def EIIP(seq):
    table = {"A": 0.12601, "T": 0.13400, "C": 0.08060, "G": 0.13350}
    return np.array([table[x] for x in seq])

def numerical_transform(seq):
    table = {"A": 0, "G": 1, "C": 2, "T": 3}
    return np.array([table[x] for x in seq])

def NAC(seq):
    cnt = Counter(seq); L = len(seq)
    return np.array([cnt[b]/L for b in ['A','C','G','T']])

def DNC(seq):
    nts = ['A','C','G','T']
    di = [''.join(p) for p in product(nts, repeat=2)]
    cnt = {k:0 for k in di}
    for i in range(len(seq)-1):
        s = seq[i:i+2]
        if s in cnt: cnt[s]+=1
    total = max(1, len(seq)-1)
    return np.array([cnt[k]/total for k in di])

def TNC(seq):
    nts = ['A','C','G','T']
    tri = [''.join(p) for p in product(nts, repeat=3)]
    cnt = {k:0 for k in tri}
    for i in range(len(seq)-2):
        s = seq[i:i+3]
        if s in cnt: cnt[s]+=1
    total = max(1, len(seq)-2)
    return np.array([cnt[k]/total for k in tri])

def CKSNAP(seq, k=2):
    nts = ['A','C','G','T']
    pairs = [''.join(p) for p in product(nts, repeat=2)]
    feat = []
    for gap in range(k+1):
        pc = {p:0 for p in pairs}; tot=0
        for i in range(len(seq)-gap-1):
            p = seq[i] + seq[i+gap+1]
            if p in pc: pc[p]+=1; tot+=1
        if tot==0: feat.extend([0]*len(pairs))
        else: feat.extend([pc[p]/tot for p in pairs])
    return np.array(feat)

physicochemical_properties = {
    'A':[1,0,0,0,0.5,0.5,0.5],
    'C':[0,1,0,0,0.0,0.5,0.3],
    'G':[0,0,1,0,0.5,1.0,0.1],
    'T':[0,0,0,1,0.5,0.0,0.2],
}
def NCP(seq):
    feats=[]
    for ch in seq:
        feats.extend(physicochemical_properties.get(ch,[0]*7))
    return np.array(feats)

def featurize_sequence(seq):
    # 逐序列提取若干种统计/理化特征并拼接
    f_num  = numerical_transform(seq)
    f_eiip = EIIP(seq)
    f_nac  = NAC(seq)
    f_dnc  = DNC(seq)
    f_tnc  = TNC(seq)
    f_ck   = CKSNAP(seq, k=2)
    f_ncp  = NCP(seq)
    # 拼接为一维
    return np.concatenate([f_num, f_eiip, f_nac, f_dnc, f_tnc, f_ck, f_ncp], axis=0)

def _load_sequences(dataset_name, split):
    assert dataset_name=='Dataset_mouse'
    base = os.path.join(root_dir, 'data/Dataset_mouse/npy')
    pos = np.load(os.path.join(base, f'{split}_seq_positive.npy')).tolist()
    neg = np.load(os.path.join(base, f'{split}_seq_negative.npy')).tolist()
    seqs = np.concatenate([pos, neg], axis=0)
    return seqs

def Bio_feature_out(dataset_name, seq_length=41, out_dim=24):
    # 加载 train/test 全部序列
    train_seqs = _load_sequences(dataset_name, 'train')
    test_seqs  = _load_sequences(dataset_name, 'test')

    # 提特征
    train_feats = np.array([featurize_sequence(s) for s in train_seqs])  # [N_train, F]
    test_feats  = np.array([featurize_sequence(s) for s in test_seqs])   # [N_test,  F]

    # 标准化
    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    test_feats  = scaler.transform(test_feats)

    # 把一维特征映射成 [L, out_dim] 的“序列化表示”
    # 简单做法：线性投影到 L*out_dim，然后 reshape 为 [L, out_dim]
    in_dim = train_feats.shape[1]
    W = np.random.randn(in_dim, seq_length * out_dim).astype(np.float32) / np.sqrt(in_dim)
    b = np.zeros((seq_length * out_dim,), dtype=np.float32)

    def project(arr):
        mat = arr @ W + b                 # [N, L*out_dim]
        mat = mat.reshape(-1, seq_length, out_dim)  # [N, L, out_dim]
        return mat

    train_final = project(train_feats)
    test_final  = project(test_feats)

    return train_final, test_final
