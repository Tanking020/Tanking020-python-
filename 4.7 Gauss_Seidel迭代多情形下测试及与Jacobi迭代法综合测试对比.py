import numpy as np
import time


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000, verbose=False):  # 默认False，需要的时候再打开
    n = len(A)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()  # 建立副本保护原始数据
    is_diag_dominant = True
    for i in range(n):
        diag_val = abs(A[i, i])
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag_val <= row_sum:
            is_diag_dominant = False
            break  # 一种很好的思路，先假设严格对角占优成立，然后进行检验，只要有一行不符合就否定

    if verbose and is_diag_dominant:
        print("矩阵是严格对角占优的，gauss_seidel迭代保证收敛")

    # 开始迭代
    for k in range(max_iter):
        x_old = x.copy()  # 保存上一次迭代的解用于计算误差

        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += A[i, j] * x[j]
                # j < i（左边）	已经填好的格子，直接抄过来
                # j = i（自己）	正在填的格子，要计算
                # j > i（右边）	还没填到的格子，用上次草稿的值

            sum2 = 0
            for j in range(i + 1, n):
                sum2 += A[i, j] * x_old[j]
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        error = np.max(np.abs(x - x_old))

        if verbose and k % 50 == 0:
            print(f"迭代{k}:误差 = {error:.6e}")

        if error < tol:
            if verbose:
                 print(f"在{k + 1}次迭代后收敛，最终误差 = {error:.6e}")
            return x, k + 1, True

    if verbose:
        print(f"警告：在{max_iter}次迭代后仍未达到收敛容差")
    return x, max_iter, False


def test_gauss_seidel():  # 开始测试迭代方法
    print("=" * 60)
    print("gauss_seidel 迭代法测试")
    print("=" * 60)

    # 例1：严格对角占优矩阵（保证收敛）
    print("\n测试用例1：严格对角占优矩阵")
    print("-" * 40)

    A1 = np.array([[10, 2, 1],
                   [1, 10, 2],
                   [2, 1, 10]], dtype=float)

    b1 = np.array([13, 13, 13], dtype=float)

    x_true1 = np.array([1, 1, 1], dtype=float)

    start_time = time.time()
    x1, iter1, converged1 = gauss_seidel(A1, b1, tol=1e-10, max_iter=1000, verbose=True)
    elapsed1 = time.time() - start_time

    # 计算误差
    error1 = np.linalg.norm(x1 - x_true1)
    print(f"\n真实解：{x_true1}")
    print(f"计算解：{x1}")
    print(f"绝对误差：{error1:.2e}")
    print(f"计算时间：{elapsed1:.6f}秒")
    print(f"是否收敛：{converged1}")

    # 验证解的正确性
    residual1 = np.linalg.norm(np.dot(A1, x1) - b1)
    print(f"残差范数：{residual1:.2e}")

    # if __name__ == "__main__":(如果只写一个例题这里就以这两句话直接结束)
    # test_gauss_seidel()#注意：到这段话之前都是对gauss_seidel函数的定义，最后这两行是调用执行

    # 测试用例2：非对角占优但收敛的矩阵
    print("\n" + "=" * 60)
    print("测试用例2：非对角占优但对称正定的矩阵")
    print("-" * 40)

    # 创建一个对称正定矩阵（guass-seidel迭代收敛的另一个充分条件）
    A2 = np.array([[4, 1, 0],
                   [1, 4, 1],
                   [0, 1, 4]], dtype=float)

    b2 = np.array([5, 6, 5], dtype=float)
    x_true2 = np.array([1, 1, 1], dtype=float)

    start_time = time.time()
    x2, iter2, converged2 = gauss_seidel(A2, b2, tol=1e-10, max_iter=1000, verbose=False)
    elapsed2 = time.time() - start_time

    error2 = np.linalg.norm(x2 - x_true2)
    print(f"\n真实解：{x_true2}")
    print(f"计算解：{x2}")
    print(f"绝对误差：{error2:.2e}")
    print(f"迭代次数：{iter2}")
    print(f"计算时间：{elapsed2:.6f}秒")  # 这里新学到一个.6f是保留六位小数并以浮点数记录，与.2e对应

    # 测试用例3：使用一个病态矩阵（条件数condA巨大）（可能不收敛或者收敛慢）
    print("\n" + "=" * 60)
    print("测试用例3：病态矩阵测试")
    print("-" * 40)

    # 此处创建一个病态矩阵
    A3 = np.array([[1, 2],
                   [2, 4.0001]], dtype=float)
    b3 = np.array([3, 6.0001], dtype=float)

    print("矩阵条件数：", np.linalg.cond(A3))

    x3, iter3, converged3 = gauss_seidel(A3, b3, tol=1e-8, max_iter=100, verbose=False)

    if converged3:
        print(f"迭代收敛，迭代次数：{iter3}")
        print(f"计算解：{x3}")

        # 使用numpy库求解
        x3_np = np.linalg.solve(A3, b3)
        print(f"numpy解：{x3_np}")
        error3 = np.linalg.norm(x3 - x3_np)
        print(f"与numpy解的差异：{error3:.2e}")
    else:
        print("警告，迭代未收敛或收敛缓慢")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


def example_usage():
    """使用实例"""

    print("\n" + "=" * 60)
    print("gauss_seidel迭代法使用实例")
    print("=" * 60)

    # 定义线性方程组
    A = np.array([[16, 3],
                  [7, -11]], dtype=float)

    b = np.array([11, 13], dtype=float)
    print(f"系数矩阵A:\n{A}")
    print(f"右侧向量b:{b}")

    # 使用gauss_seidel求解
    x, iterations, converged = gauss_seidel(
        A, b,
        x0=np.array([0, 0], dtype=float),
        tol=1e-8,
        max_iter=100,
        verbose=True
    )

    print(f"\n计算结果：")
    print(f"解向量x:{x}")
    print(f"迭代次数：{iterations}")
    print(f"是否收敛：{converged}")

    # 验证结果
    residual = np.dot(A, x) - b
    print(f"残差向量：{residual}")
    print(f"残差范数：{np.linalg.norm(residual):.2e}")

    # 与numpy算出来的结果进行比较
    x_np = np.linalg.solve(A, b)
    print(f"\nnumpy.linalg.solve 结果：{x_np}")
    print(f"两种方法差异：{np.linalg.norm(x - x_np):.2e}")


#if __name__ == "__main__":
    # 运行测试
    #test_gauss_seidel()

    # 运行使用实例
    #example_usage()

    #上面的代码经检验运行成功，从此处我开始尝试学习将gauss_seidel迭代与jacobi迭代做对比

def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000, verbose=False):
    """雅可比迭代法"""
    n = len(A)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()
    x_new = np.zeros(n)

    for k in range(max_iter):
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x[j]
            x_new[i] = s / A[i, i]

        error = np.linalg.norm(x_new - x)

        if verbose and k % 50 == 0:
            print(f"jacobi迭代{k}: 误差 = {error:.6e}")

        if error < tol:
            if verbose:
                print(f"jacobi在{k + 1}次迭代后收敛，误差 = {error:.6e}")
            return x_new, k + 1, True

        x = x_new.copy()

    if verbose:
        print(f"jacobi警告：在{max_iter}次迭代后仍未收敛")
    return x, max_iter, False


def compare_methods():
    """对比 jacobi 和 gauss_seidel 迭代法的性能"""
    print("\n" + "=" * 60)
    print("jacobi迭代法与gauss_seidel迭代法性能对比")
    print("=" * 60)

    #创建一个中等规模的严格对角占优矩阵
    n = 20  #矩阵大小（维数）
    np.random.seed(42)  #固定随机种子，确保结果可重复

    # 生成随机矩阵并使其严格对角占优
    A = np.random.rand(n, n) * 10
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1  #对角线占优

    #生成真实解（随机向量）
    x_true = np.random.rand(n)
    b = A @ x_true  #根据真实解计算b

    print(f"\n矩阵大小: {n} x {n}")
    print(f"矩阵条件数: {np.linalg.cond(A):.2e}")
    print("-" * 40)

    #测试jacobi迭代
    print("\njacobi迭代法")
    start_time = time.time()
    x_jacobi, iter_jacobi, converged_jacobi = jacobi(A, b, tol=1e-8, max_iter=5000, verbose=False)
    elapsed_jacobi = time.time() - start_time

    error_jacobi = np.linalg.norm(x_jacobi - x_true)

    print(f"迭代次数: {iter_jacobi}")
    print(f"计算时间: {elapsed_jacobi:.6f} 秒")
    print(f"绝对误差: {error_jacobi:.2e}")
    print(f"是否收敛: {converged_jacobi}")

    #测试 gauss_seidel迭代
    print("\ngauss_seidel迭代法】")
    start_time = time.time()
    x_gs, iter_gs, converged_gs = gauss_seidel(A, b, tol=1e-8, max_iter=5000, verbose=False)
    elapsed_gs = time.time() - start_time

    error_gs = np.linalg.norm(x_gs - x_true)

    print(f"迭代次数: {iter_gs}")
    print(f"计算时间: {elapsed_gs:.6f} 秒")
    print(f"绝对误差: {error_gs:.2e}")
    print(f"是否收敛: {converged_gs}")

    #对比总结
    print("\n" + "-" * 40)
    print("对比总结")
    print(f"gauss_seidel比jacobi快: {elapsed_jacobi / elapsed_gs:.2f} 倍")
    print(f"gauss_seidel迭代次数是jacobi的: {iter_jacobi / iter_gs:.2f} 倍")

    if error_jacobi < error_gs:
        print(f"jacobi精度更高: {error_gs / error_jacobi:.2e} 倍")
    else:
        print(f"gauss_seidel精度更高: {error_jacobi / error_gs:.2e} 倍")

    print("=" * 60)


def example_with_comparison():
    """带对比的完整示例"""
    # 原有的 example_usage
    example_usage()

    # 添加性能对比
    compare_methods()


if __name__ == "__main__":
    test_gauss_seidel()

    # 运行带对比的使用实例
    example_with_comparison()

#通过拓展代码我学习到了随机矩阵的创建办法以及算法时间对比的思路，有利于以后做算法优化的判断
