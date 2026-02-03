import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from sklearn.metrics.pairwise import cosine_similarity
import os
from tqdm import tqdm

# ==========================================
# 0. 配置 (路径与参数)
# ==========================================
# 原始数据 (用于画图回溯，必须加载)
CSV_PATH = r'D:\BTC_ACE\history_data\BTC-USDT_1M_2021_2025.csv'

# 模型权重
MODEL_PATH = r"D:\BTC_ACE\ace_trading\LSTM\models\btc_lstm_attn_model_2024-06-01.pth"

# [关键] 已存在的向量数据库路径 (直接读取，不重新生成)
DB_PATH = r"D:\BTC_ACE\ace_trading\LSTM\models\vector_db_base_2024-06-01.npy"
INDICES_PATH = r"D:\BTC_ACE\ace_trading\LSTM\models\vector_indices_base_2024-06-01.npy"

SPLIT_DATE = '2024-06-01' # 2025年作为测试集

# 模型参数 (需与训练一致)
INPUT_WIN = 30    
PRED_WIN = 5      
INPUT_DIM = 5     
EMBED_DIM = 128   
HIDDEN_DIM = 512  # [Upgraded] Increased capacity
NUM_LAYERS = 3    # [Upgraded] Deeper network

# INPUT_WIN = 30
# PRED_WIN = 5
# INPUT_DIM = 5
# EMBED_DIM = 128
# HIDDEN_DIM = 256  # 之前修正后的参数
# NUM_LAYERS = 2    # 之前修正后的参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ==========================================
# 1. 模型定义 (保持一致)
# ==========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1) 
        context = torch.sum(weights * lstm_output, dim=1)      
        return context

class CompositeModel(nn.Module):
    def __init__(self):
        super(CompositeModel, self).__init__()
        self.encoder_lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS, batch_first=True, dropout=0.0)
        self.attention = Attention(HIDDEN_DIM)
        self.encoder_fc = nn.Linear(HIDDEN_DIM, EMBED_DIM)
        
        self.recon_lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=1, batch_first=True)
        self.recon_fc = nn.Linear(HIDDEN_DIM, INPUT_DIM)
        self.recon_steps = INPUT_WIN
        
        self.pred_lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, num_layers=1, batch_first=True)
        self.pred_fc = nn.Linear(HIDDEN_DIM, INPUT_DIM)
        self.pred_steps = PRED_WIN
        
    def forward(self, x):
        lstm_out, _ = self.encoder_lstm(x)
        context = self.attention(lstm_out)
        emb = self.encoder_fc(context)
        
        emb_expanded_recon = emb.unsqueeze(1).repeat(1, self.recon_steps, 1)
        out_recon, _ = self.recon_lstm(emb_expanded_recon)
        recon = self.recon_fc(out_recon)
        
        emb_expanded_pred = emb.unsqueeze(1).repeat(1, self.pred_steps, 1)
        out_pred, _ = self.pred_lstm(emb_expanded_pred)
        pred = self.pred_fc(out_pred)
        
        return recon, pred, emb

# ==========================================
# 2. 核心加载逻辑 (混合模式)
# ==========================================
def load_resources():
    # A. 加载全量原始数据 (为了画图，必须有 OHLCV)
    print(f"1. Loading Raw Data from {CSV_PATH}...")
    cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = pd.read_csv(CSV_PATH, usecols=cols)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 构建一个时间戳到索引的映射，用于快速回溯画图
    # set_index 后查找速度是 O(1)
    df_lookup = df.set_index('datetime')
    
    # B. 加载历史向量库 (直接从硬盘读取)
    print(f"2. Loading Pre-computed Vector DB...")
    db_hist = np.load(DB_PATH)
    t_hist = np.load(INDICES_PATH, allow_pickle=True)
    print(f"   -> History DB Loaded. Shape: {db_hist.shape}")
    print(f"   -> Time Indices Loaded. Range: {t_hist[0]} ~ {t_hist[-1]}")
    
    # C. 准备 2025 测试数据
    print(f"3. Preparing 2025 Test Data...")
    mask_2025 = df['datetime'] >= SPLIT_DATE
    df_test = df[mask_2025].reset_index(drop=True)
    print(f"   -> Test Data Rows: {len(df_test)}")
    
    return df, df_lookup, db_hist, t_hist, df_test

# ==========================================
# 3. 2025数据实时推理 (仅处理测试集)
# ==========================================
def process_test_data(df_test, model):
    """
    对 2025 的数据进行归一化和 Embedding
    """
    print("4. Vectorizing 2025 Test Data (On-the-fly)...")
    data = df_test[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
    
    # 1. 预处理 (与训练一致)
    data[:, 4] = np.log1p(data[:, 4])
    
    total_win = INPUT_WIN + PRED_WIN
    windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=(total_win, 5))
    windows = windows.squeeze(axis=1)
    
    # 2. 归一化
    price_win = windows[:, :, :4]
    hist_close = price_win[:, :INPUT_WIN, 3:4]
    p_mean = np.mean(hist_close, axis=1, keepdims=True)
    p_std = np.std(hist_close, axis=1, keepdims=True) + 1e-3
    norm_price = (price_win - p_mean) / p_std
    
    vol_win = windows[:, :, 4:5]
    hist_vol = vol_win[:, :INPUT_WIN, :]
    v_mean = np.mean(hist_vol, axis=1, keepdims=True)
    v_std = np.std(hist_vol, axis=1, keepdims=True) + 1e-3
    norm_vol = (vol_win - v_mean) / v_std
    
    norm_windows = np.concatenate([norm_price, norm_vol], axis=2)
    
    # X: History, Y: Future
    X = torch.from_numpy(norm_windows[:, :INPUT_WIN, :])
    Y = torch.from_numpy(norm_windows[:, INPUT_WIN:, :])
    
    # 3. 推理生成向量
    model.eval()
    X_gpu = X.to(device)
    embeddings = []
    
    with torch.no_grad():
        # Batch size 16384 for inference
        for i in tqdm(range(0, len(X), 16384), desc="Inferencing 2025"):
            bx = X_gpu[i : i + 16384]
            _, _, emb = model(bx)
            embeddings.append(emb.cpu().numpy())
            
    db_test = np.concatenate(embeddings, axis=0)
    
    # 记录每个向量对应的时间 (窗口结束时间)
    # df_test 的第 (i + INPUT_WIN - 1) 行是第 i 个窗口的结束点
    valid_indices = np.arange(INPUT_WIN - 1, len(data) - PRED_WIN)
    t_test = df_test.iloc[valid_indices]['datetime'].values
    
    return X, Y, db_test, t_test

# ==========================================
# 4. 执行流程
# ==========================================
# A. 初始化
model = CompositeModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("✅ Model Loaded.")

# B. 加载资源
df_all, df_lookup, db_hist, t_hist, df_test = load_resources()

# C. 处理 2025
X_test, Y_test, db_test, t_test = process_test_data(df_test, model)

# ==========================================
# 5. 可视化函数 (修正版)
# ==========================================
def analyze_market_situation(test_idx, top_k=3):
    # 1. 准备 Query 数据
    query_time = pd.to_datetime(t_test[test_idx])
    print(f"\n🔍 Analyzing 2025 Market @ {query_time}")
    
    # 获取 Query 向量
    query_vec = db_test[test_idx].reshape(1, -1)
    
    # 2. 搜索 (Search)
    sim_scores = cosine_similarity(query_vec, db_hist).flatten()
    
    # Top K Similar
    best_idxs = sim_scores.argsort()[::-1][:top_k]
    best_scores = sim_scores[best_idxs]
    
    # Top K Dissimilar
    worst_idxs = sim_scores.argsort()[:top_k]
    worst_scores = sim_scores[worst_idxs]
    
    # 3. 准备绘图数据
    # 获取模型对未来的预测 (需要跑一次前向传播)
    with torch.no_grad():
        bx = X_test[test_idx:test_idx+1].to(device)
        _, pred_out, _ = model(bx)
        pred_fut_norm = pred_out[0, :, 3].cpu().numpy() 
        
    # 获取 Query 的归一化历史与真实未来
    q_hist_norm = X_test[test_idx, :, 3].numpy()
    q_fut_real_norm = Y_test[test_idx, :, 3].numpy()
    q_vol_norm = X_test[test_idx, :, 4].numpy()
    
    # 4. 开始绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 1, 2]})
    
    # --- Plot 1: 价格形态 (历史回溯 + 预测) ---
    ax = axes[0]
    ax.plot(range(INPUT_WIN), q_hist_norm, 'b-', linewidth=3, label='2025 Current (History)')
    ax.plot(range(INPUT_WIN, INPUT_WIN+PRED_WIN), q_fut_real_norm, 'b--', linewidth=2, label='2025 Real Future')
    ax.plot(range(INPUT_WIN, INPUT_WIN+PRED_WIN), pred_fut_norm, 'cyan', linestyle='-.', linewidth=2, label='Model Prediction')
    
    colors = ['#FF5733', '#33FF57', '#FF33FF']
    
    for i, idx in enumerate(best_idxs):
        # [关键修复]
        # t_hist[idx] 是 numpy.datetime64 类型，有时直接索引会报错
        # 我们先把它转为 pandas Timestamp
        hist_time = pd.Timestamp(t_hist[idx])
        
        # [关键修复] 
        # 使用 df_lookup (时间索引) 来查找行号，而不是 df_all (整数索引)
        try:
            end_pos = df_lookup.index.get_loc(hist_time)
        except KeyError:
            print(f"⚠️ Warning: Time {hist_time} not found in lookup. Skipping.")
            continue
            
        # 计算切片范围
        start_pos = end_pos - INPUT_WIN + 1
        fut_end_pos = end_pos + PRED_WIN
        
        # 使用行号去 df_all (原始数据) 切片
        raw_slice = df_all.iloc[start_pos : fut_end_pos + 1]
        raw_vals = raw_slice['Close'].values
        
        # 现场归一化 (Zero-Base Alignment for visualization)
        # 将历史数据的起点对齐到 Query 的均值，或者直接画 Z-Score
        # 这里我们重新计算 Z-Score 以便与 Query 对比
        h_hist = raw_vals[:INPUT_WIN]
        h_fut = raw_vals[INPUT_WIN:]
        
        if len(h_hist) == 0: continue
            
        mu = np.mean(h_hist)
        sigma = np.std(h_hist) + 1e-3
        
        h_hist_norm = (h_hist - mu) / sigma
        h_fut_norm = (h_fut - mu) / sigma
        
        label_txt = f"Hist: {hist_time.date()} (Sim: {best_scores[i]:.3f})"
        ax.plot(range(INPUT_WIN), h_hist_norm, color=colors[i], alpha=0.7, label=label_txt)
        ax.plot(range(INPUT_WIN, INPUT_WIN+PRED_WIN), h_fut_norm, color=colors[i], linestyle=':', linewidth=2)

    ax.set_title(f"Market Analogies @ {query_time}", fontsize=12, fontweight='bold')
    ax.axvline(x=INPUT_WIN, color='k', linestyle=':', label='Now')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: 量能结构 ---
    ax = axes[1]
    ax.fill_between(range(INPUT_WIN), 0, q_vol_norm, color='blue', alpha=0.2, label='2025 Volume')
    
    for i, idx in enumerate(best_idxs):
        hist_time = pd.Timestamp(t_hist[idx])
        end_pos = df_lookup.index.get_loc(hist_time) # 修复
        raw_slice = df_all.iloc[end_pos - INPUT_WIN + 1 : end_pos + 1]
        
        v_raw = np.log1p(raw_slice['Volume'].values)
        v_mu = np.mean(v_raw)
        v_sigma = np.std(v_raw) + 1e-3
        v_norm = (v_raw - v_mu) / v_sigma
        
        ax.plot(range(INPUT_WIN), v_norm, color=colors[i], linewidth=1.5)
        
    ax.set_title("Volume Structure (Log-Normalized)", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 3: 不相似行情 ---
    ax = axes[2]
    ax.plot(range(INPUT_WIN), q_hist_norm, 'b-', linewidth=3, label='2025 Current')
    
    colors_bad = ['gray', 'brown', 'olive']
    for i, idx in enumerate(worst_idxs):
        hist_time = pd.Timestamp(t_hist[idx])
        end_pos = df_lookup.index.get_loc(hist_time) # 修复
        raw_slice = df_all.iloc[end_pos - INPUT_WIN + 1 : end_pos + 1]
        
        h_hist = raw_slice['Close'].values
        h_norm = (h_hist - np.mean(h_hist)) / (np.std(h_hist) + 1e-3)
        
        ax.plot(range(INPUT_WIN), h_norm, color=colors_bad[i], linestyle='-', alpha=0.6, label=f"Dissim ({worst_scores[i]:.3f})")
        
        
    ax.set_title("Dissimilar Patterns (Structural Opposites)", fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 重新运行随机测试
print("\n--- 🏁 Re-starting Random Validations ---")
for _ in range(3):
    rand_i = random.randint(0, len(X_test) - 1)
    analyze_market_situation(rand_i)