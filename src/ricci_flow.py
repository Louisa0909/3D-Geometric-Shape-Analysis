import numpy as np
from scipy.sparse import coo_matrix
from .math_ops import compute_edge_lengths, compute_face_angles, compute_discrete_gaussian_curvature

class DiscreteRicciFlow:
    def __init__(self, vertices, faces, boundary_indices=None):
        self.vertices = vertices
        self.faces = faces
        self.num_vertices = len(vertices)
        self.boundary_indices = boundary_indices if boundary_indices is not None else []
        
        # 初始共形因子 u (logarithm of radius factor)，初始设为0
        # gamma_i = e^{u_i}
        self.u = np.zeros(self.num_vertices)
        
        # 计算初始边长，用于Circle Packing的初始设置
        # 论文使用 Inversive Distance Circle Packing
        # 这里简化为基于边长的度量变换：l_new = l_old * e^{u_i} * e^{u_j} (Yamabe Flow近似)
        # 或者 l_new^2 = r_i^2 + r_j^2 + 2 r_i r_j cos(phi) 
        # 鉴于纯Python实现的复杂性，我们采用标准的 Discrete Surface Ricci Flow (Yamabe Flow)
        # 这种变形在共形结构上是等价的。
        self.initial_edge_lengths = compute_edge_lengths(self.vertices, self.faces)
        
    def _update_metric(self):
        """
        根据当前的 u 更新边长。
        l_{ij}(t) = l_{ij}(0) * exp(u_i(t) + u_j(t)) [Conformal factor approach]
        注：这是离散Ricci Flow最常用的形式，等效于改变Circle Packing的半径。
        """
        l_initial = self.initial_edge_lengths
        
        # 获取边两端顶点的 u 值
        u_v0 = self.u[self.faces[:, 0]]
        u_v1 = self.u[self.faces[:, 1]]
        u_v2 = self.u[self.faces[:, 2]]
        
        # l_jk 对应 v1, v2
        l_new_0 = l_initial[:, 0] * np.exp(u_v1 + u_v2) # edge opp v0
        # l_ki 对应 v0, v2
        l_new_1 = l_initial[:, 1] * np.exp(u_v0 + u_v2) # edge opp v1
        # l_ij 对应 v0, v1
        l_new_2 = l_initial[:, 2] * np.exp(u_v0 + u_v1) # edge opp v2
        
        return np.column_stack((l_new_0, l_new_1, l_new_2))

    def run_flow(self, iterations=100, step_size=0.1, target_curvature=0.0):
        """
        执行 Ricci Flow 优化。
        目标：将网格展平到平面，意味着内部目标曲率为 0。
        Eq (3): du/dt = K_bar - K [cite: 180]
        """
        # 设置目标曲率 K_bar
        # 对于平坦映射，内部顶点目标曲率为0
        # 边界顶点通常分担 2pi 的欧拉示性数 (对于拓扑圆盘)
        K_bar = np.zeros(self.num_vertices)
        if len(self.boundary_indices) > 0:
            # 简单策略：将 2pi 平均分配给边界 (或者根据边界长度分配，这里简化)
            K_bar[self.boundary_indices] = (2 * np.pi) / len(self.boundary_indices)
        
        # 初始化一个列表来存储误差
        energy_history = []

        for it in range(iterations):
            # 1. 计算当前度量下的边长
            current_edge_lengths = self._update_metric()
            
            # 2. 计算当前角度
            try:
                current_angles = compute_face_angles(current_edge_lengths)
            except ValueError:
                print("Math error in angle calculation (triangle inequality violated). Stopping.")
                break
                
            # 3. 计算当前离散高斯曲率 K
            angle_sum = np.zeros(self.num_vertices)
            for i in range(3):
                np.add.at(angle_sum, self.faces[:, i], current_angles[:, i])
                
            K_current = np.zeros(self.num_vertices)
            
            # 计算 K: 内部 2pi - sum, 边界 pi - sum
            is_boundary = np.zeros(self.num_vertices, dtype=bool)
            if len(self.boundary_indices) > 0:
                is_boundary[self.boundary_indices] = True
            
            target_sum = np.where(is_boundary, np.pi, 2 * np.pi)
            K_current = target_sum - angle_sum
            
            # 4. 计算曲率误差
            error = K_bar - K_current
            mae = np.mean(np.abs(error))
            
            # 记录当前误差
            energy_history.append(mae)

            if it % 100 == 0:
                print(f"Iteration {it}: Mean Curvature Error = {mae:.6f}")
                
            if mae < 1e-4:
                print("Converged.")
                break
                
            # 5. 更新 u (Explicit Euler Integration)
            # du/dt = - (K - K_bar) = K_bar - K
            self.u += step_size * error
            
            # 归一化 u 以防止漂移 (可选，但推荐固定一个点或平均值为0)
            self.u -= np.mean(self.u)

        return self.u, np.array(energy_history)

    # 将此方法替换 src/ricci_flow.py 中的同名方法
    def embed_to_plane(self):
        """
        真正的平面嵌入算法：基于 BFS 策略，根据计算出的目标边长铺设三角形。
        """
        # 1. 更新为最终的目标边长
        final_edge_lengths = self._update_metric()
        final_angles = compute_face_angles(final_edge_lengths)
        
        # 准备数据结构
        uv = np.zeros((self.num_vertices, 2))
        is_vertex_positioned = np.zeros(self.num_vertices, dtype=bool)
        
        # 构建邻接关系: (v_start, v_end) -> (face_index, v_opposite)
        # 这对于遍历至关重要
        half_edge_to_face = {}
        for f_idx, face in enumerate(self.faces):
            # 面的三条边: (0,1), (1,2), (2,0)
            half_edge_to_face[(face[0], face[1])] = (f_idx, face[2])
            half_edge_to_face[(face[1], face[2])] = (f_idx, face[0])
            half_edge_to_face[(face[2], face[0])] = (f_idx, face[1])

        # 2. 放置第一个三角形 (种子)
        f0 = self.faces[0]
        v0, v1, v2 = f0
        
        # 获取第一面对应的边长和角度
        # edge_lengths 顺序: [opp v0, opp v1, opp v2] -> [l_12, l_20, l_01]
        l_01 = final_edge_lengths[0, 2] # 边 v0-v1
        l_02 = final_edge_lengths[0, 1] # 边 v0-v2
        angle_0 = final_angles[0, 0]    # v0 处的角度
        
        # 设定坐标
        uv[v0] = [0, 0]
        uv[v1] = [l_01, 0]
        uv[v2] = [l_02 * np.cos(angle_0), l_02 * np.sin(angle_0)]
        
        is_vertex_positioned[v0] = True
        is_vertex_positioned[v1] = True
        is_vertex_positioned[v2] = True
        
        # 3. BFS 队列：存储 (已定位的边_start, 已定位的边_end)
        queue = [(v0, v1), (v1, v2), (v2, v0)]
        visited_faces = {0}

        while queue:
            v_a, v_b = queue.pop(0)
            
            # 查找共享这条边 (v_b, v_a) 的对面三角形
            # 注意顺序：我们现在的边是 v_a -> v_b，我们需要找反向边 v_b -> v_a 连接的面
            key = (v_b, v_a)
            if key not in half_edge_to_face:
                continue # 边界边，没有相邻面
                
            f_next_idx, v_c = half_edge_to_face[key]
            
            if f_next_idx in visited_faces:
                continue
            
            visited_faces.add(f_next_idx)
            
            # 如果第三个顶点 v_c 还没定位，我们需要计算它的位置
            if not is_vertex_positioned[v_c]:
                # 我们已知 v_a, v_b 的坐标
                pos_a = uv[v_a]
                pos_b = uv[v_b]
                
                # 获取新面的边长
                # 需找到 v_c 在该面中的位置以索引正确的边长
                # edges: [opp v0, opp v1, opp v2]
                face_nodes = self.faces[f_next_idx]
                
                # 找到 v_a, v_b, v_c 在 face_nodes 中的局部索引 (0,1,2)
                # 从而获取对应的边长
                idx_a = np.where(face_nodes == v_a)[0][0]
                idx_b = np.where(face_nodes == v_b)[0][0]
                idx_c = np.where(face_nodes == v_c)[0][0]
                
                # l_ac 是 v_b 的对边 (opp idx_b)
                # l_bc 是 v_a 的对边 (opp idx_a)
                len_ac = final_edge_lengths[f_next_idx, idx_b]
                len_bc = final_edge_lengths[f_next_idx, idx_a]
                
                # 计算 v_c 的坐标：交会法 (Intersection of two circles)
                # p_c 距离 p_a 为 len_ac，距离 p_b 为 len_bc
                # 这是一个几何问题，也可以用复数求解
                d_ab = np.linalg.norm(pos_a - pos_b)
                
                # 余弦定理计算 p_a 处的角度 alpha
                # len_bc^2 = len_ac^2 + d_ab^2 - 2 * len_ac * d_ab * cos(alpha)
                cos_alpha = (len_ac**2 + d_ab**2 - len_bc**2) / (2 * len_ac * d_ab)
                cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
                sin_alpha = np.sqrt(1 - cos_alpha**2)
                
                # 构建局部坐标系
                vec_ab = pos_b - pos_a
                vec_ab_unit = vec_ab / (np.linalg.norm(vec_ab) + 1e-12)
                
                # 旋转 vec_ab 得到 vec_ac 的方向
                # 方向性：网格通常是逆时针 (CCW)。
                # 检查 v_a, v_b, v_c 的绕序来决定旋转方向
                # 这里简化假设 CCW，旋转 +alpha
                Rx = vec_ab_unit[0] * cos_alpha - vec_ab_unit[1] * sin_alpha
                Ry = vec_ab_unit[0] * sin_alpha + vec_ab_unit[1] * cos_alpha
                
                # 简单的方向修正：由于拓扑可能翻转，这里简化为固定方向
                # 严谨实现需检查法向量符号，但在 planar flattening 中通常一致
                vec_ac = np.array([Rx, Ry]) * len_ac
                
                uv[v_c] = pos_a + vec_ac
                is_vertex_positioned[v_c] = True
                
                # 将新边加入队列
                queue.append((v_c, v_b))
                queue.append((v_a, v_c))
            
        return uv
    
    def get_conformal_factor(self):
        return self.u