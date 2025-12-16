import sys
import os
import traceback
import torch
import numpy as np
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
from torch_geometric.data import Batch

# === 1. 导入模型定义 ===
# 假设 model2.py 在同一目录下
from model2 import model

# === 2. 导入特征提取依赖 ===
# 需要确保 feature_extract 文件夹在当前目录下
from feature_extract.BERT import tokenizer, bert_model, dna_to_text
from feature_extract.BDGraph import build_dna_pyg_graph, global_pca, fit_global_pca, PCA_MODEL_PATH
from feature_extract.Bio_feature import featurize_sequence


# 配置
class Config:
    out_channels = 24
    gnn_hidden = 48
    trans_layers = 4
    trans_nhead = 8
    drop_gnn = 0.3
    drop_fuse = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径配置
    model_path = "model_save.pth"
    threshold_path_bal = "best_threshold_bal.npy"
    threshold_path_mcc = "best_threshold_mcc.npy"

    # 序列参数
    seq_length = 41
    bert_dim = 24
    bio_dim = 24


args = Config()
app = Flask(__name__)
CORS(app)


# ==========================================
# 辅助函数：单条序列特征提取适配器
# ==========================================

def get_bert_single(sequence, seq_len=41, out_dim=24):
    """适配 BERT.py：将单条序列转为 BERT 特征 [1, 41, 24]"""
    text = dna_to_text(sequence)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                           max_length=seq_len + 2, truncation=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        outputs = bert_model(**inputs)
        token_embeddings = outputs.last_hidden_state[:, 1:-1, :]  # [1, L, 768]

        # Padding / Truncating
        if token_embeddings.size(1) < seq_len:
            pad = torch.zeros(1, seq_len - token_embeddings.size(1), token_embeddings.size(2), device=args.device)
            token_embeddings = torch.cat([token_embeddings, pad], dim=1)
        elif token_embeddings.size(1) > seq_len:
            token_embeddings = token_embeddings[:, :seq_len, :]

        # 降维 768 -> 24 (分组平均)
        group_size = token_embeddings.size(2) // out_dim
        reduced = []
        for i in range(out_dim):
            s = i * group_size
            e = (i + 1) * group_size if i < out_dim - 1 else token_embeddings.size(2)
            reduced.append(token_embeddings[:, :, s:e].mean(dim=2, keepdim=True))
        reduced_embeddings = torch.cat(reduced, dim=2)  # [1, L, 24]

    return reduced_embeddings


def get_bio_single(sequence, seq_len=41, out_dim=24):
    """适配 Bio_feature.py：提取理化特征并投影"""
    # 1. 提取原始特征 (一维)
    raw_feats = featurize_sequence(sequence)  # numpy array

    # 2. 模拟 Bio_feature_out 中的投影逻辑
    # ⚠️ 警告：为了保证一致性，这里的随机种子必须与训练时一致
    np.random.seed(42)  # 假设训练使用了 seed=42

    # 重新生成投影矩阵 W 和 b
    in_dim = raw_feats.shape[0]
    W = np.random.randn(in_dim, seq_len * out_dim).astype(np.float32) / np.sqrt(in_dim)
    b = np.zeros((seq_len * out_dim,), dtype=np.float32)

    # 3. 投影
    feat_projected = raw_feats @ W + b
    feat_reshaped = feat_projected.reshape(1, seq_len, out_dim)  # [1, 41, 24]

    return torch.tensor(feat_reshaped, dtype=torch.float, device=args.device)


def get_graph_single(sequence):
    """适配 BDGraph.py：构建 PyG 图数据"""
    if not os.path.exists(PCA_MODEL_PATH):
        print("Warning: PCA model not found. Fitting a temporary one.")
        fit_global_pca([sequence] * 10, n_components=8)
    else:
        from feature_extract.BDGraph import global_pca
        if global_pca is None:
            with open(PCA_MODEL_PATH, 'rb') as f:
                import pickle
                from feature_extract import BDGraph
                BDGraph.global_pca = pickle.load(f)

    # 构建图
    data = build_dna_pyg_graph(sequence)
    batch_data = Batch.from_data_list([data])
    return batch_data.to(args.device)


# ==========================================
# 初始化：加载模型和阈值
# ==========================================

print(f"Loading model on {args.device}...")
model_instance = model(
    out_channels=args.out_channels,
    gnn_hidden=args.gnn_hidden,
    trans_layers=args.trans_layers,
    trans_nhead=args.trans_nhead,
    drop_gnn=args.drop_gnn,
    drop_fuse=args.drop_fuse
).to(args.device)

# 加载权重
if os.path.exists(args.model_path):
    state_dict = torch.load(args.model_path, map_location=args.device)
    model_instance.load_state_dict(state_dict, strict=False)
    model_instance.eval()
    print("Model weights loaded successfully.")
else:
    print(f"ERROR: {args.model_path} not found!")

# 加载阈值
THRESHOLD = 0.5
if os.path.exists(args.threshold_path_bal):
    THRESHOLD = float(np.load(args.threshold_path_bal))
    print(f"Loaded Balanced Threshold: {THRESHOLD}")
elif os.path.exists(args.threshold_path_mcc):
    THRESHOLD = float(np.load(args.threshold_path_mcc))
    print(f"Loaded MCC Threshold: {THRESHOLD}")


# ==========================================
# API 路由
# ==========================================

@app.route('/test_connection', methods=['GET'])
def test():
    return "Deep Learning Model API is Online."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 获取数据
        data = request.json
        sequence = data.get("sequence", "").upper()  # 确保大写

        if not sequence or not all(c in "ATCGN" for c in sequence):
            return jsonify({'error': 'Invalid sequence format. Must be DNA string.'}), 400

        # 2. 特征预处理
        x1 = get_bert_single(sequence, args.seq_length, args.bert_dim)
        x3 = get_bio_single(sequence, args.seq_length, args.bio_dim)
        graph_batch = get_graph_single(sequence)

        # 3. 模型推理
        with torch.no_grad():
            logits = model_instance(x1, graph_batch, x3)
            prob = torch.sigmoid(logits).item()

        # 4. 结果格式化
        result = 1 if prob >= THRESHOLD else 0

        response = {
            'sequence_length': len(sequence),
            'prediction': result,
            'prediction_class': "Positive" if result == 1 else "Negative"
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345

    print(f"Starting server on port {port}...")
    app.run(port=port, debug=False, use_reloader=False)