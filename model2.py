import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import RGCNConv, global_mean_pool

# ---- 位置编码 ----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=41):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---- RGCN 模块：使用边类型 edge_type（0/1/2）----
class RGCNModule(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, num_rels=3, dropout=0.2):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden, num_relations=num_rels)
        self.conv2 = RGCNConv(hidden,   out_channels, num_relations=num_rels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type, batch):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = global_mean_pool(x, batch)  # [B, out_channels]
        return x

# ---- 主模型 ----
class model(nn.Module):
    """
    三分支（x1 序列 / x2 图 / x3 Bio）+ 融合
    x1：DNABERT 24 维 token 表示 -> 可训练投影 + Transformer
    x2：图 RGCN(使用 edge_type)
    x3：Bio 特征（线性适配到 24 维）
    返回 logits（未过 Sigmoid）
    """
    def __init__(self,
                 out_channels=24,
                 gnn_hidden=48,
                 trans_layers=4,
                 trans_nhead=8,
                 drop_gnn=0.3,
                 drop_fuse=0.5):
        super().__init__()

        # === x1: DNABERT token 表示 [B, L, 24] ===
        self.x1_proj = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 24)
        )
        self.positional_encoding = PositionalEncoding(d_model=24)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=24, nhead=trans_nhead, dim_feedforward=96, dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=trans_layers)
        self.output_projection = nn.Linear(24, 24)

        # === x2: Graph with edge types ===
        # BDGraph.py 中节点特征为 8 维（PCA 后）
        self.gnn = RGCNModule(in_channels=8, hidden=gnn_hidden, out_channels=out_channels, num_rels=3, dropout=drop_gnn)
        self.gnn_projection = nn.Linear(out_channels, 41 * 24)  # 展平到 [B, 41, 24] 以便三分支对齐

        # === x3: Bio 特征线性适配到 24 维（更稳，不易过拟合） ===
        self.x3_proj = nn.Linear(24, 24)

        # === 融合：concat 后展平 ===
        # 三分支对齐为 [B, 41, 24] -> concat dim=1 => [B, 123, 24] -> 展平
        fused_dim = 123 * 24
        self.fc1 = nn.Linear(fused_dim, 160)
        self.fc2 = nn.Linear(160, 1)
        self.dropout = nn.Dropout(drop_fuse)

    def forward(self, x1_seq, graph_batch, x3_bio):
        """
        x1_seq: [B, L, 24]
        graph_batch: PyG Batch (x, edge_index, edge_attr(edge_type), batch)
        x3_bio: [B, L, 24]
        """
        # x1
        x1 = self.x1_proj(x1_seq)                   # [B, L, 24]
        x1 = self.positional_encoding(x1)
        x1 = self.transformer_encoder(x1)
        x1 = self.output_projection(x1)             # [B, L, 24]

        # x2
        x, edge_index, edge_type, batch = (
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr.squeeze(1),
            graph_batch.batch
        )
        x2_graph = self.gnn(x, edge_index, edge_type, batch)  # [B, out_channels]
        x2 = self.gnn_projection(x2_graph).view(-1, 41, 24)   # [B, 41, 24]

        # x3
        x3 = self.x3_proj(x3_bio)                   # [B, L, 24]

        # 融合
        fused = torch.cat([x1, x2, x3], dim=1)      # [B, 123, 24]
        fused = fused.reshape(fused.size(0), -1)    # [B, 123*24]
        fused = self.dropout(F.relu(self.fc1(fused)))
        logits = self.fc2(fused).squeeze(1)         # [B]   （返回 logits）
        return logits
