#总结：此次任务实现了：
    # 1.对系数矩阵进行分析判断迭代方法是否一定收敛
    # 2.对测试方程组的三类迭代法分别求解（达到设定迭代次数上限仍未收敛的报警告）
    # 3.J-G-S三类线性方程组迭代方法的误差（迭代增量）的收敛建图分析
    # 4.J-G-S三类迭代方法的时间测量与最短时间方法建表分析

    #建立元组意识！！！key= lambda x ： x[i] 要理解
    #学会怎么建立图/表和调整各项参数

import numpy as np
import time
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','PingFang SC','Heiti SC']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告

#Jacobi迭代法
def Jacobi_with_history(A, b, x0=None, tol=1e-10, max_iter=1000, verbose=False):
    n = len(A)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()
    if verbose:
        print("\n" + "=" * 60)
        print("Jacobi迭代：")
        print("=" * 60)
    is_diag_dominant = True
    for i in range(n):
        sum1 = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= sum1:
            is_diag_dominant = False
            break

    if verbose and is_diag_dominant:
        print("这个矩阵是严格对角占优矩阵，Jacobi迭代收敛")

    x_new = np.zeros(n)
    errors = []#记录误差历史！！！

    for k in range(max_iter):
        for i in range(n):
            sum1 = b[i] - sum(x[j] * A[i, j] for j in range(n) if j != i)
            x_new[i] = sum1 / A[i, i]
        error = np.linalg.norm(x_new - x)
        errors.append(error)#记录！！！

        if error < tol:
            if verbose:
                print(f"此次Jacobi迭代过程在第{k + 1}次迭代时候迭代增量={error:.6e}在容差内，Jacobi迭代解为{x_new}")
            return x_new,k+1,True,errors
        x = x_new.copy()
    if verbose:
        print(f"\n警告：Jacobi迭代法在第{max_iter}次迭代后迭代增量仍未达到允许容差")
    return x ,max_iter,False, errors

#Gauss-Seidel迭代法
def Gauss_Seidel_with_history(A, b, x0=None, tol=1e-10, max_iter=1000, verbose=False):  # 默认False，需要的时候再打开
    n = len(A)
    if x0 is None:
        x0 = np.zeros(n)

    x = x0.copy()  # 建立副本保护原始数据
    if verbose:
        print("\n" + "=" * 60)
        print("Gauss_Seidel迭代：")
        print("=" * 60)
    is_diag_dominant = True
    for i in range(n):
        diag_val = abs(A[i, i])
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag_val <= row_sum:
            is_diag_dominant = False
            break  # 一种很好的思路，先假设严格对角占优成立，然后进行检验，只要有一行不符合就否定

    if verbose and is_diag_dominant:
        print("\n矩阵是严格对角占优的，Gauss_Seidel迭代保证收敛")

    # 开始迭代
    errors = []
    for k in range(max_iter):
        x_old = x.copy()  # 保存上一次迭代的解用于计算误差

        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += A[i, j] * x[j]

            sum2 = 0
            for j in range(i + 1, n):
                sum2 += A[i, j] * x_old[j]
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        error = np.linalg.norm(x - x_old)
        errors.append(error)

        if verbose and k % 50 == 0:
            print(f"Gauss_Seidel迭代{k}次后误差 = {error:.6e}")

        if error < tol:
            if verbose:
                print(f"Gauss_Seidel在{k + 1}次迭代后收敛，迭代增量 = {error:.6e}，Gauss_Seidel迭代解为{x}")
            return x, k + 1, True , errors

    if verbose:
        print(f"警告：Gauss_Seidel迭代在{max_iter}次迭代后仍未达到收敛容差!!!")
    return x, max_iter, False , errors

#SOR松弛法
def SOR_with_history(A, b, x0=None, w=None , tol=1e-10, verbose=False, max_iter=1000):  # 开始定义SOR函数
    if verbose:
        print("\n" + "=" * 60)
        print("SOR迭代：")
        print("=" * 60)
        print()
    n = len(A)  # 标记矩阵维数（解向量维数）
    if x0 is None:
        x0 = np.zeros(n)  # 在x0空值时定为0向量初值
    if w is None:
        w = 1.0

    if verbose:
        if 0 < w < 2:
            print(f"松弛因子为{w}，满足松弛法条件1")
        else:
            print(f"松弛因子为{w},不满足松弛法条件1，可能不收敛")

    x = x0.copy()
    is_diag_dominant = True  # 先假设成立，遍历中发现任一反例立马推翻
    for i in range(n):
        diag_val = abs(A[i, i])
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)  # 核心语句，学习到了求和运算以及遍历和判断
        if diag_val <= row_sum:
            is_diag_dominant = False
            break

    if verbose:
        print(f"此方程组的系数矩阵是否是严格对角占优矩阵(对应松弛法条件2)：{is_diag_dominant}")

    if verbose:
        if 0 < w < 2 and is_diag_dominant:  # is_diag_dominant 本身就是 True/False，不需要再比较(易错！！！)
            print("SOR松弛法必定收敛")
        else:
            print("SOR松弛法可能不收敛")

    errors = []
    for k in range(max_iter):  # (这里接上了之前的if条件判断)
        x_old = x.copy()  # 这里创建副本的意义一定要理解

        for i in range(n):
            sum1 = 0
            for j in range(i):
                sum1 += A[i, j] * x[j]

            sum2 = 0
            for j in range(i + 1, n):
                sum2 += A[i, j] * x_old[j]
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
            x[i] = (w) * x[i] + (1 - w) * x_old[i]

        error = np.linalg.norm(x - x_old)
        errors.append(error)

        if verbose and k % 10 == 0:
            print(f"第{k+1}次SOR迭代的误差为 = {error:.6e}")

        if error < tol:
            if verbose:
                print(f"在第{k+1}次迭代后SOR迭代增量= {error:.6e}进入容许值，结束迭代过程，松弛法解为{x}")
            return x, k + 1, True , errors

    if verbose:
        print(f"警告，SOR迭代在第{max_iter}次后迭代增量仍未收敛至容差内！！！")

    return x, max_iter, False , errors

#此函数用于绘制三种迭代法的收敛曲线对比图
def plot_convergence_comparison(A,b):
    print("\n正在运行Jacobi迭代法")
    _,_,_,err_j = Jacobi_with_history(A,b, verbose=True)
    print("\n正在运行Gauss_Seidel迭代法")
    _,_,_,err_gs = Gauss_Seidel_with_history(A,b, verbose=True)
    print("\n正在运行SOR迭代法")
    _,_,_,err_sor = SOR_with_history(A,b,w=1.2, verbose=True)#如果需要解或迭代次数就去掉_，_,相当于无视这个变量

    plt.figure(figsize=(10,6))

    plt.plot(err_j,'o-',label = 'Jacobi',linewidth=1.5,markersize=4)
    plt.plot(err_gs,'s-',label = 'Gauss_Seidel',linewidth=1.5,markersize=4)
    plt.plot(err_sor,'^-',label = 'SOR(w=1.2)',linewidth=1.5,markersize=4)

    plt.xlabel('迭代次数',fontsize=12)
    plt.ylabel('误差',fontsize=12)
    plt.title('三种迭代法收敛曲线对比（线性坐标）',fontsize=14)
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.savefig('convergence_linear.png',dpi=150,bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,6))

    plt.semilogy(err_j,'o-',label = 'Jacobi',linewidth=1.5,markersize=4)
    plt.semilogy(err_gs, 's-', label='Gauss-Seidel', linewidth=1.5, markersize=4)
    plt.semilogy(err_sor, '^-', label='SOR (w=1.2)', linewidth=1.5, markersize=4)

    plt.xlabel('迭代次数',fontsize=12)
    plt.ylabel('误差（对数坐标）',fontsize=12)
    plt.title('三种迭代法收敛曲线对比（半对数坐标）',fontsize=14)
    plt.legend()
    plt.grid(True,alpha=0.3)
    plt.savefig('convergence_log.png',dpi=150,bbox_inches='tight')
    plt.show()

#额外再写一个对比时间并叙述迭代法针对测试矩阵优劣的函数
def compare_time_J_G_S(A,b,w=1.2):
    """详细对比三种迭代法性能"""
    print("\n" + "=" * 60)
    print("详细性能对比")
    print("=" * 60)

    results = []#建立初始的空元组

    #Jacobi时间传出
    start = time.time()
    _,iter_j,conv_j,_ = Jacobi_with_history(A,b, verbose=False)
    time_j = time.time() - start
    results.append(("Jacobi",time_j))

    #Gauss时间传出
    start = time.time()
    _,iter_gs,conv_gs,_ = Gauss_Seidel_with_history(A,b, verbose=False)
    time_gs = time.time() - start
    results.append(("Gauss_Seidel",time_gs))

    #SOR时间传出
    start = time.time()
    _,iter_sor,conv_sor,_ = SOR_with_history(A,b, w=1.2, verbose=False)
    time_sor = time.time() - start
    results.append(("SOR(w=1.2)",time_sor))#创建元组（名字，时间）#注意（（））写成一个参数

    #注意  :<12表示左对齐并且占据12宽度
    #开始打印表格

    #输出结果
    print("\n总耗时对比（秒）：")
    print("-" * 40)
    for name , t in results:
        print(f"{name:15} : {t:.6f}秒")

    #找出最快的方法
    fastest = min(results, key=lambda x: x[1])#key=是按什么标准比较，x[1]是返回元组的第二项（数字）（第一项是名字）
    #一句话：元组让名字和时间绑定在一起永不分离，lambda x: x[1]只是临时取出时间做比较，比较完后返回的仍然是完整的元组。
    print("\n" + "=" * 40)
    print(f"最快的是：{fastest[0]}迭代法")
    print("=" * 40)

    print(f"\n迭代次数对比：")
    print(f"Jacobi: {iter_j} 次")
    print(f"Gauss-Seidel: {iter_gs} 次")
    print(f"SOR(w=1.2): {iter_sor} 次")

    return results


if __name__ == "__main__":
    #测试方程组
    A = np.array([[5,1,2],
                  [1,4,1],
                  [2,1,6]],dtype=float)
    b = np.array([10,8,16],dtype=float)

    plot_convergence_comparison(A,b)
    compare_time_J_G_S(A, b, w=1.2)