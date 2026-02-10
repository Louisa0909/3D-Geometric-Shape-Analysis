import numpy as np

def compute_probability_distribution(feature_values, num_bins=25):
    """
    计算特征值的概率分布。
    对应论文公式 (7) [cite: 243]。
    
    Input:
        feature_values: 1D array of scalar values (Conformal Factor or Mean Curvature)
        num_bins: number of intervals (论文推荐 25 或 20)
    """
    # 移除 NaN 或 Inf
    valid_values = feature_values[np.isfinite(feature_values)]
    
    if len(valid_values) == 0:
        return np.zeros(num_bins)

    # 使用 numpy 计算直方图
    # density=False 返回计数，density=True 返回概率密度
    # 论文公式 (7) P_i = n_i / n
    counts, _ = np.histogram(valid_values, bins=num_bins)
    
    total_vertices = len(valid_values)
    probabilities = counts / total_vertices
    
    return probabilities

def compute_shannon_entropy(probabilities):
    """
    计算香农熵。
    对应论文公式 (8)[cite: 248]: H = - sum(p * log(p))
    """
    # 过滤掉 0 概率以避免 log(0)
    p = probabilities[probabilities > 0]
    
    if len(p) == 0:
        return 0.0
        
    # 论文未指定底数，通常信息论用 log2 (bits) 或 ln (nats)。
    # 鉴于公式 (8) 一般形式，我们使用自然对数或 log2。
    # 代码使用 log2。
    entropy = -np.sum(p * np.log2(p))
    
    return entropy

def calculate_surface_entropy(feature_vector, bins=25):
    """
    封装函数：直接从特征向量计算熵值。
    """
    probs = compute_probability_distribution(feature_vector, num_bins=bins)
    entropy = compute_shannon_entropy(probs)
    return entropy