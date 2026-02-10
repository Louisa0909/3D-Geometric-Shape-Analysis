import numpy as np

def compute_edge_lengths(vertices, faces):
    """
    计算网格所有边的长度。
    返回: edges (M, 3) 对应每个面的三条边长 [e_ij, e_jk, e_ki]
    """
    # 获取三角形的三个顶点坐标
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 计算边长 (欧几里得距离)
    l01 = np.linalg.norm(v1 - v0, axis=1) # edge opposite to v2
    l12 = np.linalg.norm(v2 - v1, axis=1) # edge opposite to v0
    l20 = np.linalg.norm(v0 - v2, axis=1) # edge opposite to v1

    # 按照顶点索引顺序返回: [l12, l20, l01] -> 对应 v0, v1, v2 的对边
    # 注意：为了后续计算角度方便，通常存储为:
    # edge lengths associated with face [l_jk, l_ki, l_ij]
    return np.column_stack((l12, l20, l01))

def compute_face_angles(edge_lengths):
    """
    利用余弦定理计算每个面的三个内角。
    Input: edge_lengths (M, 3) -> [a, b, c] 对应面的三边
    Output: angles (M, 3) -> [angle_at_v0, angle_at_v1, angle_at_v2]
    """
    a = edge_lengths[:, 0] # opp v0
    b = edge_lengths[:, 1] # opp v1
    c = edge_lengths[:, 2] # opp v2

    # 余弦定理: a^2 = b^2 + c^2 - 2bc cos(A)
    # cos(A) = (b^2 + c^2 - a^2) / 2bc
    cos_a = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_b = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_c = (a**2 + b**2 - c**2) / (2 * a * b)

    #防止数值误差导致的 NaN
    cos_a = np.clip(cos_a, -1.0, 1.0)
    cos_b = np.clip(cos_b, -1.0, 1.0)
    cos_c = np.clip(cos_c, -1.0, 1.0)

    angles = np.column_stack((np.arccos(cos_a), np.arccos(cos_b), np.arccos(cos_c)))
    return angles

def compute_vertex_areas(vertices, faces):
    """
    计算顶点的对偶面积（通常取邻接三角形面积的1/3）。
    用于公式 (5) 和 (6) 中的 area(B)。
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # 叉乘计算三角形面积的一半
    cross_prod = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    
    num_vertices = len(vertices)
    vertex_areas = np.zeros(num_vertices)
    
    # 累加面积到顶点
    for i in range(3):
        np.add.at(vertex_areas, faces[:, i], face_areas / 3.0)
        
    return vertex_areas, face_areas

def compute_discrete_gaussian_curvature(vertices, faces, boundary_vertices_indices=None):
    """
    计算离散高斯曲率 K(v)。
    引用论文 Eq (2) [cite: 113]。
    
    K(v) = 2pi - sum(theta) (内部顶点)
    K(v) = pi - sum(theta) (边界顶点)
    """
    edge_lengths = compute_edge_lengths(vertices, faces)
    face_angles = compute_face_angles(edge_lengths)
    
    num_vertices = len(vertices)
    angle_sum = np.zeros(num_vertices)
    
    for i in range(3):
        np.add.at(angle_sum, faces[:, i], face_angles[:, i])
        
    K = np.zeros(num_vertices)
    
    # 默认为内部顶点 2pi
    target_sum = np.full(num_vertices, 2 * np.pi)
    
    # 如果提供了边界顶点索引，修改为 pi
    if boundary_vertices_indices is not None:
        target_sum[boundary_vertices_indices] = np.pi
        
    K = target_sum - angle_sum
    return K

def compute_dihedral_angles(vertices, faces):
    """
    计算二面角，用于平均曲率计算。
    """
    # 计算法向量
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1)[:, None] # 归一化

    # 构建边到面的映射 (Edge to Faces)
    # 这部分比较复杂，需要知道哪两个面共享一条边
    # 这里为了简洁，仅返回构建邻接关系所需的数据结构
    # 在 feature.py 中会更详细处理
    return normals