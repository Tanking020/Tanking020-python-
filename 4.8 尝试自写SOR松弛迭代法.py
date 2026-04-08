import time
import numpy as np  # 引入numpy库

def SOR(A, b, x0=None, w=None , tol=1.0e-6, verbose=False, max_iter=1000):  # 开始定义SOR函数
    n = len(A)  # 标记矩阵维数（解向量维数）
    if x0 is None:
        x0 = np.zeros(n)  # 在x0空值时定为0向量初值
    if w is None:
        w = 1.0

        # 这里应该是要开始进行矩阵运算，但我想到可能要定义一下松弛因子w(而且开始之前应该先判断收敛)
        # 超松弛法收敛一个充分条件：0＜w＜2且严格对角占优
        # ❌！！！ 下面这些代码都在 if 内部！如果 x0 有值就不会执行（修改缩进到与前面if同缩进）
    if 0 < w < 2:
        print(f"松弛因子为{w}，满足松弛法条件1")
    else:
        print(f"松弛因子为{w},不满足松弛法条件1，可能不收敛")
        # 这里我想要对A[i,j],j!=i 元素绝对值求和并且与A[i,i]元素绝对值比较
    x = x0.copy()
    is_diag_dominant = True  # 先假设成立，遍历中发现任一反例立马推翻
    for i in range(n):
        diag_val = abs(A[i, i])
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)  # 核心语句，学习到了求和运算以及遍历和判断
        if diag_val <= row_sum:
            is_diag_dominant = False
            break  # (要在判伪的情形下才结束循环)

    print(f"此方程组的系数矩阵是否是严格对角占优矩阵：{is_diag_dominant}")

    if 0 < w < 2 and is_diag_dominant:  # is_diag_dominant 本身就是 True/False，不需要再比较(易错！！！)
        print("\nSOR松弛法收敛")

        # 接下来写SOR松弛法过程
        # 第一步是先用Gauss_Seidel办法写新解
        # G-S方法是1、从上往下遍历执行 2、将已经得到的值替换为新值
        # x=x0.copy()
        # for i in range(n):
        # s = sum(A[i,j] * x[j] for j in range(n) if j!=i)
        # c = b[i]-s
        # x_new = c / A[i,i]
        # return x_new
        # 这段有明显问题，应该修改为如下代码

    for k in range(max_iter):  # (这里接上了之前的if条件判断)
        x_old = x.copy()  # 这里创建副本的意义一定要理解

        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += A[i, j] * x[j]

            sum2 = 0
            for j in range(i + 1, n):
                sum2 += A[i, j] * x_old[j]  # 注意x_old也是从0开始执行，而在经历过一次迭代后x_old被下边这几句代码赋予了值

            x[i] = (b[i] - sum1 - sum2) / A[i, i]  # 从这里下次循环到新的x_old

            # 这里进入松弛法特殊步骤（旧解与gauss-seidel迭代解进行加权平均）
            x[i] = (w) * x[i] + (1 - w) * x_old[i]  # 这段代码中对于x_old的暂存是精髓

        error = np.max(np.abs(x - x_old))

        if verbose and k % 50 == 0:
            print(f"第{k+1}次SOR迭代的误差为 = {error:.6e}")

        # 接下来根据误差判断什么时候结束循环

        if error < tol:
            if verbose:
                print(f"在第{k+1}次迭代后SOR迭代误差进入容许值，结束迭代过程，误差为 = {error:.6e}")#for in range语句k是从0开始，所以要加1

            return x, k + 1, True

    if verbose:  # 这里一定要与for k ……同缩进，表示在所有k in range(max_iter)都没实现的情况下才判断警告
        print(f"警告，SOR迭代在第{max_iter}次后仍未收敛至容差内！！！")

    return x, max_iter, False

    # 在return后整个函数立刻结束，也就是说SOR法已经书写完毕

    # 测试用例1：
def test_SOR():#定义函数并且开始往里面输入值
    print("=" * 60)
    print("测试用例1：严格对角占优矩阵A")
    print("=" * 60)
    #开始输入各项值
    A=np.array([[10,1,2],
                [2,12,4],
                [4,1,16]],dtype=float)
    b=np.array([2,4,2],dtype=float)
    x0=np.array([0,0,0],dtype=float)
    verbose=True

    x_true=np.linalg.solve(A,b)
    start_time=time.time()#计时
    x, iter, converged = SOR(A, b, x0=None, w=1.2, tol=1.0e-6, verbose=verbose, max_iter=1000)#对应SOR迭代函数返回结果,注意左边verbose是参数名，右边verbose是变量名
    elapsed = time.time() - start_time
    error = np.linalg.norm(x_true - x)

    print(f"\n真实解：{x_true}")
    print(f"计算解：{x}")
    print(f"绝对误差：{error:.2e}")
    print(f"计算时间：{elapsed:.6f}秒")
    print(f"是否收敛：{converged}")

    #接下来用系数矩阵与解向量乘积 与 b 进行比较求出残差（residual）范数判断解的正确性
    residual = np.linalg.norm(np.dot(A,x) - b)
    print(f"残差范数为{residual:.2e}")

    #最后别忘了调用函数
if __name__=="__main__":
    test_SOR()