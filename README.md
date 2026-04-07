# 2026.4.6
Jacobi迭代法测试
# My Optimization Journey (我的优化算法学习之路)

> 研0预备役 | 数学优化方向 | 记录从零实现数值线性代数与优化算法

## 📚 学习进度

| 算法 | 状态 | 实现文件 | 学习笔记 |
| :--- | :--- | :--- | :--- |
| Jacobi Iteration (雅可比迭代) | ✅ 已实现 | `jacobi.py` | 理解了对角占优与收敛性 |
| Gauss-Seidel (高斯-赛德尔) | 🚧 进行中 | `gauss_seidel.py` | 相比 Jacobi，利用了最新分量 |
| Gradient Descent (梯度下降) | 📅 计划中 | - | - |

## 🧠 核心理解（示例）

### Jacobi 迭代法
- **思想**：将线性方程组 $Ax=b$ 分解，利用第 $k$ 步的值迭代第 $k+1$ 步。
- **收敛条件**：严格对角占优矩阵必收敛。
- **代码关键点**：注意避免原地更新，需要设置 `x_new` 和 `x_old`。

## 🚀 如何运行代码

## 📝 日志
- **2026.04.06**：初始化仓库，上传 Jacobi 迭代法实现。

Gauss-Seidel迭代法测试
My Optimization Journey (我的优化算法学习之路)

研0预备役 | 数学优化方向 | 记录从零实现数值线性代数与优化算法

📚 学习进度
算法	状态	实现文件	学习笔记
Jacobi Iteration (雅可比迭代)	✅ 已完成	jacobi.py	理解了对角占优与收敛性
Gauss-Seidel (高斯-赛德尔)	✅ 已完成	gauss_seidel.py	理解了利用最新分量的加速技巧
SOR (逐次超松弛迭代)	📅 计划中	-	待学习
Gradient Descent (梯度下降)	📅 计划中	-	-
🧠 核心理解（今日重点）
Gauss-Seidel 迭代法
核心思想：与 Jacobi 同时使用上一步所有旧值不同，Gauss-Seidel 在计算新分量时，立即使用已经算出的新分量（j < i 部分），从而加快收敛速度。

数学表达：

text
x_i^(k+1) = (b_i - Σ_j<i a_ij·x_j^(k+1) - Σ_j>i a_ij·x_j^(k)) / a_ii
收敛条件：

严格对角占优矩阵 → 必收敛

对称正定矩阵 → 必收敛

代码关键点：

sum1：累加左边已更新的分量（j < i），用本轮新值 x[j]

sum2：累加右边未更新的分量（j > i），用上轮旧值 x_old[j]

收敛判断用无穷范数 np.max(np.abs(x - x_old))，比 L2 范数更快

增加 verbose 参数控制输出，兼顾调试与安静运行

⚔️ 性能对比（今日实验）
对比 Jacobi 与 Gauss-Seidel 在相同 20×20 严格对角占优矩阵上的表现：

算法	迭代次数	计算时间	收敛精度
Jacobi	312 次	0.0235 秒	9.87e-09
Gauss-Seidel	156 次	0.0082 秒	8.45e-09
结论：Gauss-Seidel 比 Jacobi 快约 2.85 倍，迭代次数减少约 50%。

📝 代码结构
text
gauss_seidel.py
├── gauss_seidel()        # 核心迭代函数
├── test_gauss_seidel()   # 测试用例（对角占优/对称正定/病态矩阵）
├── example_usage()       # 使用示例
└── compare_methods()     # Jacobi vs Gauss-Seidel 性能对比
🐛 今日 Debug 记录
错误	原因	解决方案
迭代不收敛	收敛判断误放在 for i 循环内部	移到所有分量更新完成之后
b 定义错误	写成了 2×2 矩阵	改为一维数组 [11, 13]
缩进错误	测试用例2、3 多缩进	减少4个空格，与 print 对齐
💡 今日收获
Gauss-Seidel 的本质：不是"原地更新"的 Jacobi，而是"用最新信息加速"的策略

收敛判断的位置很重要：必须等本轮所有分量都更新完再判断

verbose 参数设计：默认 False（安静运行），需要时手动打开，适合调试

无穷范数：np.max(np.abs(x - x_old)) 比 L2 范数计算更快，适合迭代法

🚀 如何运行代码
bash
python gauss_seidel.py
📋 明日计划
实现 SOR（逐次超松弛迭代）并对比收敛速度

可视化三种迭代法的误差下降曲线

开始梯度下降方法的学习

📎 日志
2026.04.06：初始化仓库，上传 Jacobi 迭代法实现

2026.04.07：完成 Gauss-Seidel 迭代法实现，添加性能对比模块，修复多个 bug
