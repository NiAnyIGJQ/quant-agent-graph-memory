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
# [修改] 读取 1小时级别数据
CSV_PATH = r'D:\BTC_ACE\history_data\BTC-USDT_4H_2021_2025.csv'

# [保持] 使用 1分钟级别训练好的模型权重
MODEL_PATH = r"D:\BTC_ACE\ace_trading\LSTM\models\btc_lstm_attn_model_2024-06-01.pth"

# 2025年作为测试集 (Query)，之前的作为历史库 (History DB)
SPLIT_DATE = '2024-06-01'

# [重要] 参数必须与训练时的模型完全一致，否则加载权重会报错
INPUT_WIN = 30    # 此时代表 30小时 (原本是30分钟)
PRED_WIN = 5      # 此时代表 5小时
INPUT_DIM = 5     
EMBED_DIM = 128   
HIDDEN_DIM = 512  # 请确保这与你训练时的设置一致
NUM_LAYERS = 3    # 请确保这与你训练时的设置一致

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ==========================================
# 1. 模型定义 (必须与训练代码完全一致)
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
# 2. 通用向量化函数 (核心修改)
# ==========================================
def vectorize_dataframe(df, model, desc="Vectorizing"):
    """
    输入: 原始 DataFrame (OHLCV)
    输出: 
      - embeddings: 向量库 (N, 128)
      - timestamps: 对应的时间戳 (N,)
      - X_tensor: 归一化后的输入张量 (用于画图)
      - Y_tensor: 归一化后的未来张量 (用于画图)
    """
    print(f"   -> Processing {len(df)} rows for {desc}...")
    
    # 1. 提取数值
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
    
    # 2. 预处理 (Log Volume)
    data[:, 4] = np.log1p(data[:, 4])
    
    # 3. 滑动窗口
    total_win = INPUT_WIN + PRED_WIN
    # stride_tricks 可能会消耗大量内存，对于1H数据量(几万行)没问题
    windows = np.lib.stride_tricks.sliding_window_view(data, window_shape=(total_win, 5))
    windows = windows.squeeze(axis=1) # Shape: (N_windows, 35, 5)
    
    if len(windows) == 0:
        print("   ⚠️ Data too short for windowing!")
        return None, None, None, None

    # 4. 窗口内 Z-Score 归一化 (关键：独立归一化)
    # 价格归一化
    price_win = windows[:, :, :4]
    hist_close = price_win[:, :INPUT_WIN, 3:4] # 只用历史 Close 计算均值方差
    p_mean = np.mean(hist_close, axis=1, keepdims=True)
    p_std = np.std(hist_close, axis=1, keepdims=True) + 1e-3
    norm_price = (price_win - p_mean) / p_std
    
    # 成交量归一化
    vol_win = windows[:, :, 4:5]
    hist_vol = vol_win[:, :INPUT_WIN, :]
    v_mean = np.mean(hist_vol, axis=1, keepdims=True)
    v_std = np.std(hist_vol, axis=1, keepdims=True) + 1e-3
    norm_vol = (vol_win - v_mean) / v_std
    
    # 合并
    norm_windows = np.concatenate([norm_price, norm_vol], axis=2)
    
    # 5. 转换为 Tensor
    X_tensor = torch.from_numpy(norm_windows[:, :INPUT_WIN, :])
    Y_tensor = torch.from_numpy(norm_windows[:, INPUT_WIN:, :])
    
    # 6. 模型推理 (生成 Embedding)
    model.eval()
    embeddings = []
    
    # 批处理推理以防显存溢出
    batch_size = 8192
    with torch.no_grad():
        for i in tqdm(range(0, len(X_tensor), batch_size), desc=desc):
            bx = X_tensor[i : i + batch_size].to(device)
            _, _, emb = model(bx)
            embeddings.append(emb.cpu().numpy())
            
    db_vectors = np.concatenate(embeddings, axis=0)
    
    # 7. 对齐时间戳 (取 Input Window 的最后一个时间点作为当前时间 T)
    # valid_indices 对应的是 windows 在原始 df 中的起始行号
    # 我们的 T 时刻是 start + INPUT_WIN - 1
    valid_indices = np.arange(INPUT_WIN - 1, len(data) - PRED_WIN)
    timestamps = df.iloc[valid_indices]['datetime'].values
    
    return db_vectors, timestamps, X_tensor, Y_tensor

# ==========================================
# 3. 数据加载与构建流程
# ==========================================
def load_and_build_db():
    # A. 加载模型
    print(f"1. Loading 1M-Model from {MODEL_PATH}...")
    model = CompositeModel().to(device)
    # 允许部分不匹配(strict=False)通常不建议，但如果Dropout层命名不同可尝试
    # 这里假设结构完全一致
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # B. 加载 1H 原始数据
    print(f"2. Loading 1H Data from {CSV_PATH}...")
    cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_all = pd.read_csv(CSV_PATH, usecols=cols)
    df_all['datetime'] = pd.to_datetime(df_all['datetime'])
    df_all = df_all.sort_values('datetime').reset_index(drop=True)
    
    # 创建快速查找索引 (用于画图回溯)
    df_lookup = df_all.set_index('datetime')
    
    # C. 切分 历史库(Train) 和 测试集(Test)
    mask_test = df_all['datetime'] >= SPLIT_DATE
    df_hist = df_all[~mask_test].reset_index(drop=True) # 2021-2024
    df_test = df_all[mask_test].reset_index(drop=True)  # 2025
    
    print(f"   -> History Data (for DB): {len(df_hist)} rows")
    print(f"   -> Query Data (for Test): {len(df_test)} rows")
    
    # D. [核心] 现场构建向量库 (Rebuilding DB in RAM)
    print("\n3. Rebuilding Vector DB from 1H History (Applying 1M Model)...")
    db_hist, t_hist, _, _ = vectorize_dataframe(df_hist, model, desc="Building History DB")
    
    print("\n4. Vectorizing 2025 Query Data...")
    db_test, t_test, X_test, Y_test = vectorize_dataframe(df_test, model, desc="Vectorizing Query")
    
    return model, df_all, df_lookup, db_hist, t_hist, db_test, t_test, X_test, Y_test

# ==========================================
# 4. 可视化函数 (保持逻辑，适配新数据源)
# ==========================================
def analyze_market_situation(model, test_idx, db_hist, t_hist, X_test, Y_test, db_test, df_all, df_lookup, top_k=3):
    # 1. 准备 Query
    query_time = pd.Timestamp(t_test[test_idx])
    print(f"\n🔍 Analyzing 2025 (1H) Market @ {query_time}")
    
    query_vec = db_test[test_idx].reshape(1, -1)
    
    # 2. 搜索相似度
    sim_scores = cosine_similarity(query_vec, db_hist).flatten()
    
    best_idxs = sim_scores.argsort()[::-1][:top_k]
    best_scores = sim_scores[best_idxs]
    
    worst_idxs = sim_scores.argsort()[:top_k]
    worst_scores = sim_scores[worst_idxs]
    
    # 3. 获取模型预测 (用于画图)
    with torch.no_grad():
        bx = X_test[test_idx:test_idx+1].to(device)
        _, pred_out, _ = model(bx)
        pred_fut_norm = pred_out[0, :, 3].cpu().numpy()
        
    q_hist_norm = X_test[test_idx, :, 3].numpy()
    q_fut_real_norm = Y_test[test_idx, :, 3].numpy()
    q_vol_norm = X_test[test_idx, :, 4].numpy()
    
    # 4. 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 1, 2]})
    
    # Plot 1: 价格结构
    ax = axes[0]
    ax.plot(range(INPUT_WIN), q_hist_norm, 'b-', linewidth=3, label='2025 Current (1H)')
    ax.plot(range(INPUT_WIN, INPUT_WIN+PRED_WIN), q_fut_real_norm, 'b--', linewidth=2, label='2025 Real Future')
    ax.plot(range(INPUT_WIN, INPUT_WIN+PRED_WIN), pred_fut_norm, 'cyan', linestyle='-.', label='Model Pred')
    
    colors = ['#FF5733', '#33FF57', '#FF33FF']
    
    for i, idx in enumerate(best_idxs):
        hist_time = pd.Timestamp(t_hist[idx])
        
        # 使用时间索引快速定位原始数据
        try:
            end_pos = df_lookup.index.get_loc(hist_time)
        except KeyError:
            continue
            
        # 切片: 前30小时 + 后5小时
        start_pos = end_pos - INPUT_WIN + 1
        fut_end_pos = end_pos + PRED_WIN
        
        # 边界检查
        if start_pos < 0 or fut_end_pos >= len(df_all): continue
            
        raw_slice = df_all.iloc[start_pos : fut_end_pos + 1]
        raw_vals = raw_slice['Close'].values
        
        # 现场归一化 (Z-Score) 以匹配 Query 的尺度
        h_hist = raw_vals[:INPUT_WIN]
        h_fut = raw_vals[INPUT_WIN:]
        
        mu = np.mean(h_hist)
        sigma = np.std(h_hist) + 1e-3
        
        h_hist_norm = (h_hist - mu) / sigma
        h_fut_norm = (h_fut - mu) / sigma
        
        label_txt = f"Hist: {hist_time.date()} (Score: {best_scores[i]:.3f})"
        ax.plot(range(INPUT_WIN), h_hist_norm, color=colors[i], alpha=0.7, label=label_txt)
        ax.plot(range(INPUT_WIN, INPUT_WIN+PRED_WIN), h_fut_norm, color=colors[i], linestyle=':', linewidth=2)
        
    ax.set_title(f"1-Hour Fractal Matching @ {query_time}", fontsize=14, fontweight='bold')
    ax.axvline(x=INPUT_WIN, color='k', linestyle=':')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 成交量结构
    ax = axes[1]
    ax.fill_between(range(INPUT_WIN), 0, q_vol_norm, color='blue', alpha=0.2, label='Current Vol')
    
    for i, idx in enumerate(best_idxs):
        hist_time = pd.Timestamp(t_hist[idx])
        try:
            end_pos = df_lookup.index.get_loc(hist_time)
            raw_slice = df_all.iloc[end_pos - INPUT_WIN + 1 : end_pos + 1]
            v_raw = np.log1p(raw_slice['Volume'].values)
            v_norm = (v_raw - np.mean(v_raw)) / (np.std(v_raw) + 1e-3)
            ax.plot(range(INPUT_WIN), v_norm, color=colors[i], linewidth=1.5, alpha=0.8)
        except: continue
            
    ax.set_title("Volume Structure (Log-Norm)", fontsize=12)
    ax.grid(True)
    
    # Plot 3: 结构对立面 (Dissimilar)
    ax = axes[2]
    ax.plot(range(INPUT_WIN), q_hist_norm, 'b-', linewidth=3, label='Current')
    for i, idx in enumerate(worst_idxs):
        hist_time = pd.Timestamp(t_hist[idx])
        try:
            end_pos = df_lookup.index.get_loc(hist_time)
            raw_vals = df_all.iloc[end_pos - INPUT_WIN + 1 : end_pos + 1]['Close'].values
            h_norm = (raw_vals - np.mean(raw_vals)) / (np.std(raw_vals) + 1e-3)
            ax.plot(range(INPUT_WIN), h_norm, color='gray', alpha=0.5, label=f"Dissim {worst_scores[i]:.2f}")
        except: continue
            
    ax.set_title("Dissimilar Structures (Pattern Opposites)", fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    # A. 加载数据并重建库
    model, df_all, df_lookup, db_hist, t_hist, db_test, t_test, X_test, Y_test = load_and_build_db()
    
    print("\n✅ Database Rebuilt in RAM.")
    print(f"   History Vectors: {db_hist.shape}")
    print(f"   Test Vectors:    {db_test.shape}")
    
    # B. 随机测试 3 次
    print("\n--- 🏁 Starting Cross-Timeframe Validations ---")
    for _ in range(3):
        # 随机选一个 2025 年的时间点
        rand_i = random.randint(0, len(X_test) - 1)
        
        # 执行分析
        analyze_market_situation(
            model, rand_i, 
            db_hist, t_hist, 
            X_test, Y_test, db_test, 
            df_all, df_lookup
        )