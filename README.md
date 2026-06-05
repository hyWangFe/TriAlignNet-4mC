# Mus4mCPred - DNA 4mC Methylation Site Prediction System

**Mus4mCPred: Accurate identification of DNA N4-methylcytosine sites in mouse genome using multi-view feature learning and deep hybrid network.**

基于深度学习的 DNA 4mC (N4-methylcytosine) 甲基化位点预测系统。采用三分支融合架构：DNABERT-6 序列特征 + 图神经网络 (RGCN) 结构特征 + 理化特征，对 41bp DNA 序列进行二分类预测。

## 项目结构

```
.
├── api.py                    # Flask API 服务（推理入口）
├── model.py                  # 模型定义（三分支融合架构）
├── train.py                  # 训练脚本（10-fold 交叉验证）
├── test.py                   # 测试脚本
├── Data_process.py           # 数据加载与预处理
├── utils.py                  # 评估工具函数
├── environment.yml           # Conda 环境配置
│
├── feature_extract/          # 特征提取模块
│   ├── BERT.py               # DNABERT-6 token embedding 提取
│   ├── BDGraph.py            # DNA 结构图构建（PyG）
│   ├── Bio_feature.py        # 理化特征提取（EIIP, NAC, DNC 等）
│   └── global_pca_model.pkl  # 预拟合的 PCA 降维模型
│
├── bert_model/               # DNABERT-6 预训练模型
│   └── dna_bert_6/           # 模型权重及配置文件
│
├── templates/                # 前端页面
│   └── index.html            # 可视化预测界面
│
├── data/                     # 数据集
│   └── Dataset_mouse/npy/    # 小鼠 4mC 数据集
│
├── model_save.pth            # 训练好的模型权重
└── best_threshold_bal.npy    # 最优分类阈值
```

## 环境要求

- **操作系统**: Windows / Linux
- **Python**: 3.9
- **CUDA**: 12.1+（GPU 推理推荐，CPU 也可运行）
- **Conda**: Miniconda 或 Anaconda

## 部署步骤

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml -n tri1
conda activate tri1
```

### 2. 安装 PyG 扩展包

PyTorch Geometric 扩展包需要特殊的 wheel 索引，需单独安装：

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

### 3. 准备模型文件

确保以下文件存在于项目根目录：

| 文件 | 说明 |
|------|------|
| `bert_model/dna_bert_6/` | DNABERT-6 预训练模型（含 pytorch_model.bin, config.json 等） |
| `model_save.pth` | 训练好的预测模型权重 |
| `best_threshold_bal.npy` | 最优分类阈值（约 0.498） |
| `feature_extract/global_pca_model.pkl` | 预拟合的 PCA 降维模型 |

### 4. 启动推理服务

```bash
python -u api.py
```

服务默认启动在 `http://127.0.0.1:12345`，可通过命令行参数修改端口：

```bash
python -u api.py 8080
```

### 5. 访问预测界面

浏览器打开 `http://127.0.0.1:12345` 即可使用可视化预测界面。

## API 接口

### 连接测试

```
GET /test_connection
```

响应：
```
Deep Learning Model API is Online.
```

### 序列预测

```
POST /predict
Content-Type: application/json
```

请求体：
```json
{
    "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGA"
}
```

参数说明：
- `sequence`: DNA 序列（推荐 41bp），仅包含 A/T/C/G/N 字符

响应示例：
```json
{
    "sequence_length": 41,
    "prediction": 1,
    "prediction_class": "Positive"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| sequence_length | int | 输入序列长度 |
| prediction | int | 预测结果 (0=Negative, 1=Positive) |
| prediction_class | string | 预测类别 |

## 模型训练（可选）

如需重新训练模型：

```bash
python train.py --seed 42 --batch_size 128 --epochs 100
```

可选参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --seed | 42 | 随机种子 |
| --dataset | Dataset_mouse | 数据集名称 |
| --batch_size | 128 | 批次大小 |
| --epochs | 100 | 训练轮数 |
| --out_channels | 24 | BERT 输出维度 |
| --gnn_hidden | 48 | GNN 隐藏层维度 |
| --trans_layers | 4 | Transformer 层数 |
| --trans_nhead | 8 | 注意力头数 |
| --learning_rate | 0.001 | 学习率 |

## 技术栈

- **深度学习框架**: PyTorch 2.5.1
- **图神经网络**: PyTorch Geometric (RGCNConv)
- **预训练语言模型**: DNABERT-6
- **Web 框架**: Flask + Flask-CORS
- **前端**: Tailwind CSS + Chart.js

## 引用

本文代码对应论文：*Mus4mCPred: Accurate identification of DNA N4-methylcytosine sites in mouse genome using multi-view feature learning and deep hybrid network.*
