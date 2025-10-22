import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
import argparse
import warnings
warnings.filterwarnings('ignore')

from Data_process import DC_labels_tensor, device
from feature_extract.BERT import Bert_out
from feature_extract.BDGraph import get_graph_datasets
from feature_extract.Bio_feature import Bio_feature_out
from model2 import model
from utils import Model_Evaluate

def random_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_seed(42)

def cal_best_threshold_balAcc(y_true, y_prob, n_steps=101):
    """扫描阈值，最大化 Balanced Accuracy = (SN+SP)/2"""
    best_thr, best_bal = 0.5, -1.0
    for thr in np.linspace(0.0, 1.0, n_steps):
        preds = (y_prob >= thr).astype(int)
        sn, sp, _, _ = Model_Evaluate(y_true, preds)
        bal = 0.5 * (sn + sp)
        if bal > best_bal:
            best_bal, best_thr = bal, thr
    return best_thr, best_bal

def train_one_epoch(model3, train_loader, graph_loader, criterion, optimizer, device, scaler, clip_norm=1.0):
    model3.train()
    all_probs, all_labels = [], []
    for (features1, features3, labels), graph_batch in zip(train_loader, graph_loader):
        features1 = torch.tensor(features1, dtype=torch.float, device=device)  # [B, L, 24]
        features3 = torch.tensor(features3, dtype=torch.float, device=device)  # [B, L, 24]
        labels    = torch.tensor(labels,    dtype=torch.float, device=device)  # [B]
        graph_batch = graph_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model3(features1, graph_batch, features3)    # [B]
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model3.parameters(), clip_norm)
        scaler.step(optimizer); scaler.update()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs); y_true = np.concatenate(all_labels)
    preds = (y_prob >= 0.5).astype(int)  # 训练过程仅监控 0.5 阈值
    sn, sp, acc, mcc = Model_Evaluate(y_true, preds)
    return float(loss.item()), acc, mcc

def validate_probs(model3, val_loader, graph_loader, device):
    model3.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for (features1, features3, labels), graph_batch in zip(val_loader, graph_loader):
            features1 = torch.tensor(features1, dtype=torch.float, device=device)
            features3 = torch.tensor(features3, dtype=torch.float, device=device)
            labels    = torch.tensor(labels,    dtype=torch.float, device=device)
            graph_batch = graph_batch.to(device)

            logits = model3(features1, graph_batch, features3)  # [B]
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
    y_prob = np.concatenate(all_probs); y_true = np.concatenate(all_labels)
    return y_true, y_prob

def main_worker(args):
    batch_size = args.batch_size

    # 图数据（训练集索引来自 train split）
    train_graph_dataset, _ = get_graph_datasets(args.dataset)

    # x1：BERT 24 维；x3：Bio 24 维
    x1_all, _ = Bert_out(args.dataset)            # [N, 41, 24]
    x3_all, _ = Bio_feature_out(args.dataset)     # [N, 41, 24]
    labels_all = DC_labels_tensor.numpy().astype(np.int64)   # [N]

    # 类不平衡：pos_weight
    pos = (labels_all == 1).sum(); neg = (labels_all == 0).sum()
    pos_weight = torch.tensor([neg / max(1, pos)], device=device)
    print(f"[INFO] pos={pos}, neg={neg}, pos_weight={float(pos_weight):.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    kf = StratifiedKFold(n_splits=args.KFold, shuffle=True, random_state=args.seed)
    best_overall_mcc, best_overall_epoch, best_overall_thr = -1, -1, 0.5
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(x1_all, labels_all)):
        print(f'\n========== 第 {fold + 1} 折 ==========')

        model3 = model(out_channels=args.out_channels,
                       gnn_hidden=args.gnn_hidden,
                       trans_layers=args.trans_layers,
                       trans_nhead=args.trans_nhead,
                       drop_gnn=args.drop_gnn,
                       drop_fuse=args.drop_fuse).to(device)

        optimizer = torch.optim.Adam(model3.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
        scaler = torch.cuda.amp.GradScaler()

        # 切分特征与标签
        tr_x1, va_x1 = x1_all[train_idx], x1_all[val_idx]
        tr_x3, va_x3 = x3_all[train_idx], x3_all[val_idx]
        tr_y,  va_y  = labels_all[train_idx], labels_all[val_idx]

        # 图子集
        train_graph_subset = torch.utils.data.Subset(train_graph_dataset, train_idx)
        val_graph_subset   = torch.utils.data.Subset(train_graph_dataset,   val_idx)

        # 可选：WeightedRandomSampler（处理类不平衡）
        if args.use_sampler:
            cls_counts = np.bincount(tr_y)
            cls_weights = 1.0 / np.maximum(1, cls_counts)
            sample_weights = cls_weights[tr_y]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            train_loader = torch.utils.data.DataLoader(
                list(zip(tr_x1, tr_x3, tr_y)),
                batch_size=batch_size, sampler=sampler, drop_last=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                list(zip(tr_x1, tr_x3, tr_y)),
                batch_size=batch_size, shuffle=True, drop_last=True
            )

        val_loader = torch.utils.data.DataLoader(
            list(zip(va_x1, va_x3, va_y)),
            batch_size=batch_size, shuffle=False, drop_last=False
        )
        train_graph_loader = DataLoader(train_graph_subset, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_graph_loader   = DataLoader(val_graph_subset,   batch_size=batch_size, shuffle=False, drop_last=False)

        best_mcc_fold, best_epoch_fold, best_thr_fold = -1, -1, 0.5
        patience, counter = args.patience, 0

        for epoch in range(args.epoch):
            print(f'---- Epoch {epoch + 1} ----')
            train_loss, train_acc, train_mcc = train_one_epoch(
                model3, train_loader, train_graph_loader, criterion, optimizer, device, scaler,
                clip_norm=args.clip_norm
            )
            scheduler.step(epoch + 1)

            y_true, y_prob = validate_probs(model3, val_loader, val_graph_loader, device)
            thr_bal, bal_at_thr = cal_best_threshold_balAcc(y_true, y_prob, n_steps=args.thr_steps)

            # 监控 0.5 阈值下的指标
            preds_monitor = (y_prob >= 0.5).astype(int)
            _sn, _sp, _acc, _mcc_mon = Model_Evaluate(y_true, preds_monitor)

            print(f'[Train] loss={train_loss:.5f} acc@0.5={_acc:.4f} mcc@0.5={_mcc_mon:.4f}')
            print(f'[Val  ] BalAcc@best={bal_at_thr:.4f} thrBAL={thr_bal:.3f} | lr={optimizer.param_groups[0]["lr"]:.6f}')

            # === 以验证集 Balanced Accuracy（在其最优阈值处）作为模型选择指标 ===
            metric_for_selection = bal_at_thr
            if metric_for_selection > best_mcc_fold:
                best_mcc_fold = metric_for_selection
                best_epoch_fold = epoch + 1
                best_thr_fold   = thr_bal
                best_model_state = model3.state_dict()
                torch.save(best_model_state, "model_save.pth")
                np.save("best_threshold_bal.npy", np.array([best_thr_fold], dtype=np.float32))
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"[Early Stop] epoch={epoch + 1}, best BalAcc={best_mcc_fold:.4f} (epoch={best_epoch_fold}, thr={best_thr_fold:.3f})")
                    break

            torch.cuda.empty_cache()

        print(f'[Fold {fold + 1}] best BalAcc={best_mcc_fold:.4f}, epoch={best_epoch_fold}, thrBAL={best_thr_fold:.3f}')
        # 记录跨折最优（可选：此处也可做平均汇报）

    print('\n>>>> 全部折完成。模型与阈值已保存：model_save.pth / best_threshold_bal.npy')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', default='Dataset_mouse')
    parser.add_argument('--batch_size', type=int, default=128)

    # 新模型可配参数（与 model2.py 对齐）
    parser.add_argument('--out_channels', type=int, default=24)
    parser.add_argument('--gnn_hidden', type=int, default=48)
    parser.add_argument('--trans_layers', type=int, default=4)
    parser.add_argument('--trans_nhead', type=int, default=8)
    parser.add_argument('--drop_gnn', type=float, default=0.3)
    parser.add_argument('--drop_fuse', type=float, default=0.5)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=60)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--KFold', type=int, default=10)
    parser.add_argument('--use_sampler', action='store_true', help='使用 WeightedRandomSampler 以缓解类不平衡')
    parser.add_argument('--clip_norm', type=float, default=1.0)
    parser.add_argument('--thr_steps', type=int, default=101, help='阈值扫描步数，建议 101 或 1001')
    args = parser.parse_args()
    main_worker(args)
