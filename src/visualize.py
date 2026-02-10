import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pyvista as pv

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    return np.load(file_path)

def plot_figure_5_energy(history, save_path=None):
    """
    复现 Figure 5: Ricci energy optimization
    展示优化过程中的误差收敛曲线。
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history, color='purple', linewidth=2, label='Curvature Error')
    
    # 模仿论文风格
    plt.title('Ricci Energy Optimization', fontsize=14)
    plt.xlabel('Iteration time', fontsize=12)
    plt.ylabel('Curvature error', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_figure_6_embedding(vertices_2d, faces, save_path=None):
    """
    复现 Figure 6: Planar Embedding
    展示展平后的 2D 网格。
    使用 PyVista, 将 2D 点作为 z=0 的 3D 点处理，并使用正交投影。
    """
    print(f"   Rendering Figure 6 with PyVista...")
    
    # 1. 数据准备：将 2D (x,y) 扩展为 3D (x,y,0)
    n_verts = vertices_2d.shape[0]
    # 拼接一列全为 0 的 z 坐标
    vertices_3d = np.hstack([vertices_2d, np.zeros((n_verts, 1))])
    
    # 2. 准备 faces (PyVista 格式: [3, v1, v2, v3, ...])
    faces_pv = np.hstack([[3] + list(f) for f in faces])
    
    # 3. 创建网格
    mesh = pv.PolyData(vertices_3d, faces_pv)
    
    # 4. 绘图配置
    pl = pv.Plotter(off_screen=True)
    pl.set_background('white') # 论文图背景通常是白色
    
    # 绘制蓝色网格线 (Wireframe 风格)
    # style='wireframe' 仅显示线条，模仿 plt.triplot
    pl.add_mesh(mesh, 
                style='wireframe', 
                color='blue', 
                line_width=1.0) # 线条宽度可调
    
    # 5. 关键设置：2D 视角
    pl.view_xy()  # 切换到 XY 平面顶视图
    pl.enable_parallel_projection() # 【关键】开启平行/正交投影，消除 3D 透视感
    
    # 6. 调整相机范围
    pl.camera.zoom(1.1) 

    # 7. 添加标题到图像上方中间
    pl.add_text("Planar Embedding", position="upper_edge", font_size=18, color="black")
    
    if save_path:
        pl.screenshot(save_path)
    pl.close()

def plot_figure_9_3d_heatmap(vertices, faces, values, title="Feature Heatmap", save_path=None):
    """
    复现 Figure 9: 3D Mesh Heatmap (关键图)
    展示 3D 海马体表面的特征分布（如 Mean Curvature 或 Conformal Factor）。
    使用 PyVista 复现。
    """
    print(f"   Rendering {title} with PyVista...")
    
    # 1. 准备数据格式
    # PyVista 的 faces 需要格式为 [点数, p1, p2, p3, 点数, p1, ...] 的扁平数组
    # 因为都是三角形，所以点数固定为 3
    faces_pv = np.hstack([[3] + list(f) for f in faces])
    
    # 2. 创建网格对象
    mesh = pv.PolyData(vertices, faces_pv)
    
    # 3. 绑定数据到顶点 (Point Data)
    # PyVista 会自动在面内进行插值，产生平滑效果
    mesh.point_data['values'] = values
    
    # 4. 设置绘图器
    # off_screen=True 表示在后台渲染，不弹出窗口，适合批量处理
    pl = pv.Plotter(off_screen=True)
    
    # 5. 添加网格到场景
    # cmap='jet': 论文同款彩虹色
    # smooth_shading=True: 开启高洛德着色(Gouraud shading)，让表面看起来光滑
    # show_edges=False: 不显示网格线，只显示颜色
    pl.add_mesh(mesh, 
                scalars='values', 
                cmap='jet', 
                smooth_shading=True, 
                show_edges=False,
                show_scalar_bar=False)
    
    # 6. 设置相机和视角
    pl.view_isometric()   # 等轴测视角
    pl.camera.zoom(1.2)   # 稍微放大一点
    
    # 7. 添加颜色条
    pl.add_scalar_bar(title="", shadow=True,
                  vertical=True,
                  width=0.05,          # 5%宽度
                  height=0.35,         # 35%高度
                  position_x=0.9,      # 距离右边10%
                  position_y=0.6)      # 从60%高度开始
    # 标题
    pl.add_text(title, position="upper_edge", font_size=18, color="black")
    
    # 8. 保存截图
    if save_path:
        pl.screenshot(save_path)
    
    # 清理资源
    pl.close()

def plot_figure_10_histograms(values, feature_name="Conformal Factor", save_path=None):
    """
    复现 Figure 10: Probability distribution histograms
    展示特征的直方图分布。
    """
    # 论文对比了不同的 bin 数量
    bins_list = [100, 40, 25, 10]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    # 去除极端值以便画图更好看
    clean_values = values[np.isfinite(values)]
    p5, p95 = np.percentile(clean_values, [1, 99])
    data_to_plot = clean_values[(clean_values > p5) & (clean_values < p95)]

    for ax, b in zip(axes, bins_list):
        # 论文风格：蓝色柱子，黑色边框
        ax.hist(data_to_plot, bins=b, color='#5D9BCF', edgecolor='black', linewidth=0.5, density=False)
        ax.set_title(f'bins = {b}', fontsize=12)
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Frequency')
        
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 1. 加载数据
    data_path = 'data/processed/bert_lh_hippocampus_results.npz'
    print(f"Loading results from {data_path}...")
    data = load_data(data_path)
    
    if data is None:
        print("请先运行 main.py 生成结果文件！")
        return

    # 提取数据
    vertices = data['vertices']
    faces = data['faces']
    vertices_2d = data['vertices_2d']
    conformal_factor = data['conformal_factor']
    mean_curvature = data['mean_curvature']
    energy_history = data['energy_history']

    # 在提取数据后，打印统计信息
    print(f"--- Data Debug Info ---")
    print(f"Conformal Factor: min={conformal_factor.min():.4f}, max={conformal_factor.max():.4f}, mean={conformal_factor.mean():.4f}")
    print(f"Mean Curvature:   min={mean_curvature.min():.4f}, max={mean_curvature.max():.4f}, mean={mean_curvature.mean():.4f}")
    
    # 检查 2D 嵌入是否失败（如果 2D 坐标全是 0 或 NaN，Conformal Factor 就会全是异常值）
    print(f"Vertices 2D:      min={vertices_2d.min():.4f}, max={vertices_2d.max():.4f}")
    
    # 创建输出目录
    output_dir = 'assets/figures_reproduced'
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 画图
    print("Plotting Figure 5 (Energy)...")
    plot_figure_5_energy(energy_history, f"{output_dir}/fig5_energy.png")
    
    print("Plotting Figure 6 (2D Embedding)...")
    plot_figure_6_embedding(vertices_2d, faces, f"{output_dir}/fig6_embedding.png")
    
    print("Plotting Figure 9 (3D Heatmaps)...")
    # Conformal Factor Heatmap
    plot_figure_9_3d_heatmap(vertices, faces, conformal_factor, 
                             "Conformal Factor Distribution (Fig 9)", 
                             f"{output_dir}/fig9_conformal_factor.png")
    # Mean Curvature Heatmap
    plot_figure_9_3d_heatmap(vertices, faces, mean_curvature, 
                             "Mean Curvature Distribution (Fig 9)", 
                             f"{output_dir}/fig9_mean_curvature.png")
    
    print("Plotting Figure 10 (Histograms)...")
    plot_figure_10_histograms(conformal_factor, "Conformal Factor", f"{output_dir}/fig10_hist_cf.png")
    plot_figure_10_histograms(mean_curvature, "Mean Curvature", f"{output_dir}/fig10_hist_mc.png")
    
    print("Done! Figures saved to assets/figures_reproduced/")

if __name__ == "__main__":
    main()