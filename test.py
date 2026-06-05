import torch
import numpy as np
import argparse
import os
import joblib  # 新增：用于加载 scaler

from model import model
from Data_process import device
from feature_extract.BERT import Bert_out
from feature_extract.Bio_feature import Bio_feature_out
from feature_extract.BDGraph import get_graph_datasets
from torch_geometric.data import DataLoader
from utils import Model_Evaluate

def load_threshold(paths, default=0.5):
    for p in paths:
        if os.path.exists(p):
            try:
                return float(np.load(p))
            except Exception:
                pass
    return default

def test(args):
    # 加载 x1/x3 原始特征
    _, x1_test = Bert_out('Dataset_mouse')
    _, x3_test = Bio_feature_out('Dataset_mouse')

    # 【修复】：加载训练集最好的 Scaler 并对测试集进行特征转换
    scaler_x1 = joblib.load("best_scaler_x1.pkl")
    scaler_x3 = joblib.load("best_scaler_x3.pkl")

    L1, D1 = x1_test.shape[1], x1_test.shape[2]
    x1_test = scaler_x1.transform(x1_test.reshape(-1, D1)).reshape(x1_test.shape[0], L1, D1)

    L3, D3 = x3_test.shape[1], x3_test.shape[2]
    x3_test = scaler_x3.transform(x3_test.reshape(-1, D3)).reshape(x3_test.shape[0], L3, D3)

    # 图数据
    _, test_graph_dataset = get_graph_datasets('Dataset_mouse')
    test_graph_loader = DataLoader(test_graph_dataset, batch_size=args.batch_size, shuffle=False)

    # 标签
    test_label_positive = np.load('data/Dataset_mouse/npy/test_label_positive.npy')
    test_label_negative = np.load('data/Dataset_mouse/npy/test_label_negative.npy')
    test_labels = np.concatenate([test_label_positive, test_label_negative], axis=0)

    test_loader = torch.utils.data.DataLoader(
        list(zip(x1_test, x3_test, test_labels)), batch_size=args.batch_size, shuffle=False
    )

    thr = load_threshold(["best_threshold_bal.npy", "best_threshold_mcc.npy"], default=0.5)
    source = "BAL" if os.path.exists("best_threshold_bal.npy") else ("MCC" if os.path.exists("best_threshold_mcc.npy") else "DEFAULT")
    print(f"[INFO] Using decision threshold ({source}): {thr:.3f}")

    test_model = model(out_channels=args.out_channels,
                       gnn_hidden=args.gnn_hidden,
                       trans_layers=args.trans_layers,
                       trans_nhead=args.trans_nhead,
                       drop_gnn=args.drop_gnn,
                       drop_fuse=args.drop_fuse).to(device)

    if not os.path.exists("model_save.pth"):
        print("[ERROR] model_save.pth not found.")
        return 0.0

    state = torch.load("model_save.pth", map_location=device)
    test_model.load_state_dict(state, strict=False)
    test_model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for (f1, f3, labels), graph_batch in zip(test_loader, test_graph_loader):
            f1 = torch.tensor(f1, dtype=torch.float, device=device)
            f3 = torch.tensor(f3, dtype=torch.float, device=device)
            labels = torch.tensor(labels, dtype=torch.float, device=device)
            graph_batch = graph_batch.to(device)

            logits = test_model(f1, graph_batch, f3)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    y_prob = np.concatenate(all_probs); y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= thr).astype(int)
    sn, sp, acc, mcc = Model_Evaluate(y_true, y_pred)
    print(f"\n[Test] SN={sn:.4f}  SP={sp:.4f}  ACC={acc:.4f}  MCC={mcc:.4f}")
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=24)
    parser.add_argument('--gnn_hidden', type=int, default=48)
    parser.add_argument('--trans_layers', type=int, default=4)
    parser.add_argument('--trans_nhead', type=int, default=8)
    parser.add_argument('--drop_gnn', type=float, default=0.3)
    parser.add_argument('--drop_fuse', type=float, default=0.5)
    args = parser.parse_args()
    test(args)