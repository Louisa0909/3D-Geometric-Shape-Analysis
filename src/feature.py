import numpy as np
from .math_ops import compute_vertex_areas, compute_edge_lengths

def compute_conformal_factor_feature(vertices_3d, faces, vertices_2d_embedded):
    """
    计算共形因子 (Conformal Factor) lambda。
    """
    # 1. 计算 3D 网格的面面积
    _, areas_3d = compute_vertex_areas(vertices_3d, faces)
    
    # 2. 计算 2D 嵌入网格的面面积
    # 2D 顶点补 0 z坐标以复用函数
    v2d_homo = np.column_stack((vertices_2d_embedded, np.zeros(len(vertices_2d_embedded))))
    _, areas_2d = compute_vertex_areas(v2d_homo, faces)
    
    # 【修复】防止 area 为 0 或 负数
    areas_3d = np.maximum(areas_3d, 1e-12)
    areas_2d = np.maximum(areas_2d, 1e-12)
    
    # 论文公式 (5): lambda = 0.5 * (log(Area_3d) - log(Area_2d))
    # 注：根据定义 e^2u * g_0 = g，即 Area_3d / Area_2d ?? 
    # 通常 conformal factor u 定义为 metric scaling: l_new = e^u * l_old
    # Area_new ~ e^2u * Area_old  =>  u = 0.5 * log(Area_new / Area_old)
    # 这里计算的是把曲面压平所需的 scaling。
    face_lambda = 0.5 * (np.log(areas_3d) - np.log(areas_2d))
    
    # 将面的值平均到顶点上
    num_vertices = len(vertices_3d)
    vertex_lambda_sum = np.zeros(num_vertices)
    vertex_face_count = np.zeros(num_vertices)
    
    for i in range(3):
        np.add.at(vertex_lambda_sum, faces[:, i], face_lambda)
        np.add.at(vertex_face_count, faces[:, i], 1)
        
    vertex_face_count = np.maximum(vertex_face_count, 1)
    lambda_v = vertex_lambda_sum / vertex_face_count
    
    # 【修复】清理任何可能的 NaN 或 Inf
    lambda_v = np.nan_to_num(lambda_v, nan=0.0, posinf=10.0, neginf=-10.0)
    
    return lambda_v

def compute_mean_curvature_feature(vertices, faces):
    """
    计算平均曲率 (Mean Curvature) H。
    """
    # 1. 构建边拓扑结构
    edge_map = {} 
    for i, f in enumerate(faces):
        # 排序确保无向边 (0,1) 和 (1,0) 是同一个 key
        edges = [tuple(sorted((f[0], f[1]))), tuple(sorted((f[1], f[2]))), tuple(sorted((f[2], f[0])))]
        for e in edges:
            if e not in edge_map: edge_map[e] = []
            edge_map[e].append(i)
            
    # 2. 计算面法向量
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1)
    norms = np.maximum(norms, 1e-12) # 防止除零
    normals /= norms[:, None]
    
    # 3. 准备累加器
    num_vertices = len(vertices)
    sum_l_beta = np.zeros(num_vertices)
    
    # 4. 遍历所有边计算 l(e) * beta(e)
    for edge, face_indices in edge_map.items():
        if len(face_indices) == 2:
            f1, f2 = face_indices
            n1, n2 = normals[f1], normals[f2]
            
            # 计算二面角 beta
            dot_prod = np.clip(np.dot(n1, n2), -1.0, 1.0)
            beta = np.arccos(dot_prod)
            
            # 计算边长 l(e)
            p1, p2 = vertices[edge[0]], vertices[edge[1]]
            length = np.linalg.norm(p1 - p2)
            
            val = length * beta
            sum_l_beta[edge[0]] += val
            sum_l_beta[edge[1]] += val
            
    # 5. 计算 1-ring 面积 (area(B))
    vertex_areas, _ = compute_vertex_areas(vertices, faces)
    
    # area(B) 近似为 3 * vertex_area
    area_B = vertex_areas * 3.0
    
    # 【修复】防止除以零：将极小面积设为 1e-9
    area_B = np.maximum(area_B, 1e-9)
    
    # 计算 H
    H_v = sum_l_beta / area_B
    
    # 【修复】去除异常值：Mean Curvature 理论上不应过大
    # 如果面积极小（退化点），H_v 会爆炸。这里将其截断或设为0
    H_v = np.nan_to_num(H_v, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 可选：截断极端值，防止统计偏差
    # H_v = np.clip(H_v, 0, 100) 
    
    return H_v