import torch
import numpy as np
import argparse
import os

from model2 import model
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
    # 加载 x1/x3
    _, x1_test = Bert_out('Dataset_mouse')        # [N_test, 41, 24]
    _, x3_test = Bio_feature_out('Dataset_mouse') # [N_test, 41, 24]

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

    # 模型与阈值：优先“均衡阈值”，回退到 MCC，再回退 0.5
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

            logits = test_model(f1, graph_batch, f3)         # [B]
            probs  = torch.sigmoid(logits).cpu().numpy()     # 概率
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

    # 与新模型参数对齐
    parser.add_argument('--out_channels', type=int, default=24)
    parser.add_argument('--gnn_hidden', type=int, default=48)
    parser.add_argument('--trans_layers', type=int, default=4)
    parser.add_argument('--trans_nhead', type=int, default=8)
    parser.add_argument('--drop_gnn', type=float, default=0.3)
    parser.add_argument('--drop_fuse', type=float, default=0.5)
    args = parser.parse_args()
    test(args)
