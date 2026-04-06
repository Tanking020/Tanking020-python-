import numpy as np


def jacobi(A, b, max_iter=100, tol=1e-10):  # "给出方程组并定义最大迭代次数和允许误差"
    n = len(b)  # "定义变量个数，即解向量长度"
    x = np.zeros(n)  # "0向量组作为初解"
    x_new = np.zeros(n)#"准备一个全是 0 的数组，用来存放即将算出的新解"

    for k in range(max_iter):  # "迭代次数计数"
        for i in range(n):  # 对每个未知数依次更新
            s = b[i]  # s存为b向量中第i个元素
            for j in range(n):  # 遍历所有变量
                if j != i:  # 跳过自己那一列
                    s -= A[i, j] * x[j]  # b[i]减去同行的j列元素，即将aiixi提出
            x_new[i] = s / A[i, i]  # 除去系数，完成迭代过程x_[n+1]=……

        # 检查收敛条件（在容差范围后停止迭代）
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged at iteration {k}")
            return x_new
        x = x_new.copy()
    print("Did not converge within max iterations")
    return x


#进行实例测试
if __name__ == "__main__":
    A = np.array([[5, 1, 2],
                  [1, 4, 1],
                  [2, 1, 6]], dtype=float)#选了一个严格对角占优阵来确保Jacobi迭代收敛
    b = np.array([10, 8, 16], dtype=float)

    x_jacobi = jacobi(A, b)
    print(f"jacobi solution:{x_jacobi}")

    x_exact = np.linalg.solve(A, b)
    print(f"Exact solution: {x_exact}")
    print(f"error:{np.linalg.norm(x_jacobi - x_exact)}")

#结果：Converged at iteration 40 经过40次迭代后收敛
#jacobi solution:[0.88659794 1.2371134  2.16494845] 雅可比迭代求出的解
#Exact solution: [0.88659794 1.2371134  2.16494845] numpy直接求出的精确解
#error:1.984351916745469e-11 误差