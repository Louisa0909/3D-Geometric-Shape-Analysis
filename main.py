import os
import numpy as np
import nibabel as nib
from src.ricci_flow import DiscreteRicciFlow
from src.feature import compute_conformal_factor_feature, compute_mean_curvature_feature
from src.entropy import calculate_surface_entropy

def find_boundary_vertices(faces):
    """
    找到网格的边界顶点索引。
    原理：边界边只属于一个三角形。
    """
    # 将所有边标准化为 (min, max) 元组
    edges = np.sort(faces[:, [0, 1]], axis=1)
    edges = np.concatenate((edges, np.sort(faces[:, [1, 2]], axis=1)), axis=0)
    edges = np.concatenate((edges, np.sort(faces[:, [2, 0]], axis=1)), axis=0)

    # 转换为 void 类型以便于使用 numpy 进行行比较
    edge_dtype = np.dtype((np.void, edges.dtype.itemsize * edges.shape[1]))
    edges_void = np.ascontiguousarray(edges).view(edge_dtype)

    # 统计每条边出现的次数
    _, idx, counts = np.unique(edges_void, return_index=True, return_counts=True)
    
    # 出现次数为 1 的边是边界边
    boundary_edges_indices = idx[counts == 1]
    boundary_edges = edges[boundary_edges_indices]
    
    # 获取边界顶点（去重）
    boundary_vertices = np.unique(boundary_edges.flatten())
    return boundary_vertices

def main():
    # 1. 设置文件路径
    data_dir = 'data'
    file_name = 'bert_lh_hippocampus.ply'
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Processing {file_name} ---")

    # 2. 加载网格数据
    # 使用 nibabel 读取 FreeSurfer 二进制几何格式
    try:
        print("Loading mesh data...")
        vertices, faces = nib.freesurfer.io.read_geometry(file_path)
        # 确保 faces 是整数类型
        faces = faces.astype(np.int32)
        print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces.")
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        print("Tip: Ensure you have installed nibabel (pip install nibabel)")
        return
    
    # ... load vertices and faces ...

    # 预处理：将封闭网格剪开一个口子，使其拓扑结构变为圆盘
    # 简单粗暴的方法：找到 Z 轴最小（或最大）的顶点，删除它及其邻域
    # 或者根据顶点索引切除一部分（需可视化确认位置）

    print("Cutting mesh to create boundary...")
    # 获取所有顶点的 Z 坐标
    z_coords = vertices[:, 2]
    
    # 策略：切掉 Z 轴底部 5% 的顶点
    # 这会形成一个大的开口（Boundary Loop），利于曲率释放
    z_threshold = np.percentile(z_coords, 5) 
    
    # 标记需要移除的顶点（Z 值小于阈值的）
    vertices_to_remove = z_coords < z_threshold
    indices_to_remove = np.where(vertices_to_remove)[0]
    
    # 找到所有包含这些顶点的面
    # np.isin 检查面的三个顶点是否在移除列表中
    mask_faces_to_remove = np.any(np.isin(faces, indices_to_remove), axis=1)
    
    # 保留剩余的面
    faces = faces[~mask_faces_to_remove]
    
    print(f"Removed {np.sum(mask_faces_to_remove)} faces (bottom 5% Z-height) to create an open surface.")
    
    # 【重要】切分后，会有很多顶点不再属于任何面（孤立顶点）
    # 必须清理它们，否则后续索引会越界或出错
    # 这里使用 trimesh 来自动清理未引用的顶点，并重新映射 faces 索引

    # 清理未引用的孤立顶点
    # 如果不清理，feature计算时会遍历到这些面积为0的孤立点，导致除零错误
    import trimesh
    temp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    # 移除未引用的顶点，并重新映射 faces 索引
    temp_mesh.remove_unreferenced_vertices()
    
    # 更新变量
    vertices = temp_mesh.vertices
    faces = temp_mesh.faces
    
    print(f"Cleaned mesh: {len(vertices)} vertices, {len(faces)} faces.")
    # ... continue to find_boundary_vertices ...

    # 3. 预处理：找到边界顶点
    # 论文提到海马体区域是开放曲面，需要处理边界条件
    print("Identifying boundary vertices...")
    boundary_indices = find_boundary_vertices(faces)
    print(f"Found {len(boundary_indices)} boundary vertices.")

    # 4. 运行 Ricci Flow 算法
    # 论文步骤 7: Optimize Ricci energy
    print("\n--- Starting Discrete Ricci Flow Optimization ---")
    rf = DiscreteRicciFlow(vertices, faces, boundary_indices)
    
    # 迭代次数和步长可能需要根据具体网格调整，这里使用典型值
    # 论文中使用牛顿法，这里我们使用梯度流（更易实现但收敛稍慢）
    target_curvature = 0.0 # 目标是将曲面展平
    _, energy_history = rf.run_flow(iterations=2000, step_size=0.005, target_curvature=0.0)
    
    # 5. 平面嵌入 (Embedding)
    # 论文步骤 8: Embed mesh in the plane
    print("Embedding mesh to 2D plane...")
    vertices_2d = rf.embed_to_plane()

    # 6. 特征提取
    # 论文步骤 9: Calculate conformal factor and mean curvature
    print("\n--- Extracting Geometric Features ---")
    
    # (1) Conformal Factor (共形因子)
    # 论文公式 (5) 
    conformal_factor = compute_conformal_factor_feature(vertices, faces, vertices_2d)
    print(f"Conformal Factor computed. Range: [{np.min(conformal_factor):.4f}, {np.max(conformal_factor):.4f}]")

    # (2) Mean Curvature (平均曲率)
    # 论文公式 (6) 
    mean_curvature = compute_mean_curvature_feature(vertices, faces)
    print(f"Mean Curvature computed. Range: [{np.min(mean_curvature):.4f}, {np.max(mean_curvature):.4f}]")

    # 7. 计算香农熵 (Shannon Entropy)
    # 论文步骤 11: Calculate entropy of lambda and H
    print("\n--- Calculating Shannon Entropy ---")
    
    # 论文中提到对于 Conformal Factor 使用 25 个 bins 效果最好 
    entropy_cf = calculate_surface_entropy(conformal_factor, bins=25)
    
    # 论文中提到对于 Mean Curvature 使用 20 个 bins 效果最好 
    entropy_mc = calculate_surface_entropy(mean_curvature, bins=20)

    print(f"\n=== Final Results for {file_name} ===")
    print(f"Conformal Factor Entropy (25 bins): {entropy_cf:.6f}")
    print(f"Mean Curvature Entropy (20 bins):   {entropy_mc:.6f}")
    
    # 输出特征向量，这可以作为后续 XGBoost 分类器的输入
    feature_vector = [entropy_cf, entropy_mc]
    print(f"Feature Vector: {feature_vector}")

    # 8. 保存数据用于可视化
    print("\n--- Saving Data for Visualization ---")
    
    # 创建保存目录
    output_dir = os.path.join(data_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义输出文件名 (例如: bert_lh_hippocampus_results.npz)
    base_name = os.path.splitext(file_name)[0]
    save_path = os.path.join(output_dir, f"{base_name}_results.npz")
    
    # 使用 numpy.savez 保存所有数组到一个文件
    np.savez(save_path,
             vertices=vertices,               # 原始 3D 顶点 (用于画 Figure 9 网格)
             faces=faces,                     # 网格面索引 (用于所有 3D/2D 绘图)
             vertices_2d=vertices_2d,         # 2D 平面坐标 (用于画 Figure 6)
             conformal_factor=conformal_factor, # 共形因子 (用于画 Figure 7, 9, 10)
             mean_curvature=mean_curvature,   # 平均曲率 (用于画 Figure 8, 9, 10)
             energy_history=energy_history    # 优化曲线 (用于画 Figure 5)
    )
    
    print(f"Data saved successfully to: {save_path}")
    print("You can now run the visualization scripts.")

if __name__ == "__main__":
    main()