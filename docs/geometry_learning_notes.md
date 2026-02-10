# 从 Ricci Flow 到几何深度学习：3D 形状分析的技术演进

## 1. 项目背景与复现工作 (The Reproduced Work)

本项目复现了 Ahmadi et al. (2024) 的工作，通过离散 Ricci Flow (Discrete Ricci Flow) 将复杂的大脑海马体（Hippocampus）3D网格曲面共形映射（Conformal Mapping）到 2D 平面。

### 核心逻辑
利用 Ricci Flow 优化目标曲率，计算出共形因子 (Conformal Factor) 和平均曲率 (Mean Curvature)。

### 创新点
不同于直接使用这些几何量，该论文引入了香农熵 (Shannon Entropy) 对几何特征进行全局编码，将其转化为紧凑的特征向量，最后使用 XGBoost 进行阿尔兹海默症的分类。

### 我的实现
- 实现了基于 Circle Packing Metric 的离散 Ricci Flow 算法。
- 计算了共形映射下的面积畸变（Conformal Factor）和曲率特征。
- 复现了特征熵的计算与可视化（如仓库中的 fig10_hist_cf.png）。

## 2. 现代几何深度学习 (Broader Perspective)

虽然 Ricci Flow 提供了坚实的数学理论基础（共形几何），但在现代 3D 视觉和图形学中，几何深度学习 (Geometric Deep Learning) 正在成为主流。通过阅读 DiffusionNet 和 HodgeNet，我对该领域的演进有了更深的理解：

### DiffusionNet (Sharp et al., 2022)
- **核心思想**：不同于 Ricci Flow 需要显式地求解参数化，DiffusionNet 引入了一个可学习的扩散层 (Diffusion Layer)。它利用热传导方程（Heat Equation）在曲面上由近及远地传播特征。
- **对比思考**：我复现的论文严重依赖网格的质量（Ricci Flow 对网格拓扑敏感），而 DiffusionNet 的最大优势是 Discretization Agnostic（离散化无关性）。它可以在不同分辨率、甚至不同拓扑（点云 vs 网格）之间迁移，这是传统几何方法难以做到的。

### HodgeNet (Smirnov & Solomon, 2021)
- **核心思想**：这篇论文从谱几何 (Spectral Geometry) 的角度切入。它不是像传统方法那样使用固定的拉普拉斯算子（Laplacian），而是通过神经网络学习 Hodge 星算子 (Hodge Star Operator) 的参数。
- **对比思考**：我复现的论文使用了固定的几何流（Ricci Flow）来提取特征。而 HodgeNet 证明了我们可以“学习”出一个算子，使得其特征向量（Eigenvectors）更适合特定的任务（如分割或分类）。这意味着我们不再受限于预定义的几何描述符。

## 3. 总结与分析 (Critical Synthesis)

通过对比这三种方法，我对 3D 几何分析有了如下总结：

| 方法类型 | 代表算法 (本项目) | 优势 (Pros) | 局限 (Cons) |
|----------|-------------------|--------------|--------------|
| 显式几何特征工程 | Ricci Flow + Entropy (当前复现) | 1. 可解释性极强：特征直接对应物理量（面积畸变、曲率）。<br>2. 理论保证：共形映射在数学上保证了角度不变性，适合医学等对形变敏感的领域。 | 1. 计算昂贵：Ricci Flow 需要迭代求解非线性方程组。<br>2. 对拓扑敏感：需要处理网格的亏格（Genus）和边界条件。 |
| 谱几何学习 | HodgeNet | 1. 端到端学习：不需要手动设计特征，直接学习算子参数。<br>2. 高效：基于稀疏矩阵运算。 | 1. 仍然依赖于网格的连接关系（Connectivity）。 |
| 空间扩散学习 | DiffusionNet | 1. 鲁棒性最强：对网格采样和分辨率变化不敏感（Robust to discretization）。<br>2. 通用性：一套架构可用于点云和网格。 | 1. 可解释性不如显式几何方法直观。 |

---

## References & Further Reading

在这篇笔记中，我精读了以下论文。本项目复现了 Paper [1] 的核心算法，并参考 Paper [2] 和 [3] 对比了传统几何方法与现代几何深度学习的差异。

**[1] The Reproduced Work (Project Core)**
> **Alzheimer's disease diagnosis by applying Shannon entropy to Ricci flow-based surface indexing and extreme gradient boosting**
> *Fatemeh Ahmadi, et al. (2024)*
> *Computer Aided Geometric Design*
> * **Role:** 本项目的核心算法来源。我复现了其中的 Discrete Ricci Flow、Curvature Calculation 以及 Entropy Feature Extraction 模块。
> [Link to Paper](https://doi.org/10.1016/j.cagd.2024.102364) (DOI)

**[2] Geometric Deep Learning (Robustness)**
> **DiffusionNet: Discretization Agnostic Learning on Surfaces**
> *Nicholas Sharp, et al. (2022)*
> *ACM Transactions on Graphics (TOG)*
> * **Insight:** 该论文提出的 DiffusionNet 解决了传统方法（如 Ricci Flow）对网格质量和拓扑结构敏感的问题，实现了“离散化无关”的学习。
> [Project Page](https://github.com/nmwsharp/diffusion-net) | [ArXiv](https://arxiv.org/abs/2012.00888)

**[3] Spectral Geometry Learning (Operator Learning)**
> **HodgeNet: Learning Spectral Geometry on Triangle Meshes**
> *Dmitriy Smirnov & Justin Solomon (2021)*
> *SIGGRAPH 2021*
> * **Insight:** 该论文展示了如何通过学习 Hodge 星算子来替代固定的几何算子，启发了我对从“人工设计特征”转向“学习几何算子”的思考。
> [Project Page](https://github.com/dmitriy-smirnov/hodgenet) | [ArXiv](https://arxiv.org/abs/2105.09508)