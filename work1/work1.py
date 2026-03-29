import sys
from pathlib import Path

import numpy as np

# 兼容两种运行方式：
# 1) 在项目根目录执行：python -m work1.work1
# 2) 直接运行文件：python work1/work1.py
# 第二种情况下，sys.path[0] 会变成 work1 目录，导致无法 import optimization，
# 所以这里显式把“项目根目录”加入 sys.path。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimization import (
    Objective,
    armijo_line_search,
    golden_ratio_line_search,
    steepest_descent,
    wolfe_powell_line_search,
)


def quadratic_difference_objective() -> Objective:
    """
    work1 目标函数：
        f(x) = (x1 - x2)^2 + x2^2
    """

    def func(x: np.ndarray) -> float:
        # f(x) = (x1-x2)^2 + x2^2
        return (x[0] - x[1]) ** 2 + x[1] ** 2

    def grad(x: np.ndarray) -> np.ndarray:
        # 手工推导梯度：
        # df/dx1 = 2*(x1-x2)
        df_dx1 = 2.0 * (x[0] - x[1])
        # df/dx2 = -2*(x1-x2) + 2*x2
        df_dx2 = -2.0 * (x[0] - x[1]) + 2.0 * x[1]
        # 返回二维梯度向量。
        return np.array([df_dx1, df_dx2], dtype=float)

    # 该作业只用到一阶信息，所以只传 func/grad。
    return Objective(func=func, grad=grad, name="quadratic_difference")


def run_line_search_experiments(x0: np.ndarray, d0: np.ndarray) -> None:
    """
    第一部分实验：固定初始点 x0 与方向 d0，比较不同线搜索策略的步长与迭代次数。
    """

    # 构造目标函数对象。
    objective = quadratic_difference_objective()
    # 设一个偏大的初始步长，便于观察回溯类方法的收缩行为。
    alpha_init = 5.0

    print("=== Part 1: line-search sensitivity on fixed x0 and d0 ===")
    print("=" * 70)
    print(f"{'Method':<15} & {'Parameter':<18} & {'Alpha':<10} & {'Iters':<5}")
    print("-" * 70)

    # 实验 1：黄金分割法，观察容差 eps 对步长精度和迭代次数的影响。
    for eps in [1e-2, 1e-4, 1e-6]:
        # 仅改变 tol，观察“精度-计算量”折中。
        alpha, count = golden_ratio_line_search(x0, d0, objective, tol=eps)
        print(f"{'GoldenRatio':<15} & eps={eps:<14} & {alpha:<10.6f} & {count:<5}")

    print("-" * 70)

    rho = 0.5
    sigma1 = 1e-4
    # 实验 2a：Armijo 法，固定 rho，比较 sigma1。
    for s1 in [1e-4, 0.1, 0.2]:
        # alpha_init 固定为较大值，便于体现回溯行为。
        alpha, count = armijo_line_search(
            x0,
            d0,
            objective,
            alpha_init=alpha_init,
            sigma1=s1,
            rho=rho,
        )
        print(f"{'Armijo':<15} & sigma1={s1:<11} & {alpha:<10.6f} & {count:<5}")

    # 实验 2b：Armijo 法，固定 sigma1，比较 rho。
    for r in [0.2, 0.5, 0.8]:
        # rho 越小，步长衰减越快，通常回溯次数更少但步长更保守。
        alpha, count = armijo_line_search(
            x0,
            d0,
            objective,
            alpha_init=alpha_init,
            sigma1=sigma1,
            rho=r,
        )
        print(f"{'Armijo':<15} & rho={r:<11} & {alpha:<10.6f} & {count:<5}")

    print("-" * 70)

    # 实验 3a：Wolfe-Powell 法，比较 sigma2。
    for s2 in [0.1, 0.5, 0.9]:
        # sigma2 越大，曲率条件通常更严格。
        alpha, count = wolfe_powell_line_search(
            x0,
            d0,
            objective,
            alpha=alpha_init,
            sigma1=1e-4,
            sigma2=s2,
        )
        print(f"{'Wolfe':<15} & sigma2={s2:<13} & {alpha:<10.6f} & {count:<5}")

    print("-" * 70)

    # 实验 3b：Wolfe-Powell 法，固定 sigma1/sigma2/rho，比较 beta。
    test_sigma1 = 0.2
    for beta in [0.1, 0.5, 2.0, 5.0]:
        # beta 作为初始搜索参考尺度，会影响 phase1/phase2 路径。
        alpha, count = wolfe_powell_line_search(
            x0,
            d0,
            objective,
            alpha=alpha_init,
            sigma1=test_sigma1,
            sigma2=0.9,
            rho=0.5,
            beta=beta,
        )
        print(f"{'Wolfe':<15} & beta={beta:<15} & {alpha:<10.6f} & {count:<5}")


def run_steepest_descent_experiments(x0: np.ndarray) -> None:
    """
    第二部分实验：将线搜索嵌入最速下降法，比较整体优化表现。
    """

    # 构造同一个目标函数，保证与第一部分实验可对照。
    objective = quadratic_difference_objective()
    # 最速下降法里每轮线搜索常用 alpha_init = 1.0 作为试探值。
    sd_alpha_init = 1.0

    print("\n=== Part 2: steepest descent with different line searches ===")
    print("=" * 80)
    print(f"{'Method':<15} & {'Parameter':<18} & {'Opt Value f(x*)':<15} & {'SD Iters':<8}")
    print("-" * 80)

    # 实验 4：最速下降 + 黄金分割法。
    # 说明：黄金分割里 tol 控制的是线搜索内层精度，不是外层停机条件。
    _, f_opt, iters, _, _ = steepest_descent(
        x0,
        objective,
        golden_ratio_line_search,
        grad_tol=1e-4,
    )
    print(f"{'SD + Golden':<15} & eps=1e-4{'':<10} & {f_opt:<15.2e} & {iters:<8}")
    print("-" * 80)

    # 实验 5：最速下降 + Armijo，比较 rho 对外层迭代次数的影响。
    for rho in [0.2, 0.5, 0.8]:
        # 只改 rho，其他参数固定，保证对比公平。
        _, f_opt, iters, _, _ = steepest_descent(
            x0,
            objective,
            armijo_line_search,
            alpha_init=sd_alpha_init,
            rho=rho,
            sigma1=1e-4,
        )
        print(f"{'SD + Armijo':<15} & rho={rho:<14} & {f_opt:<15.2e} & {iters:<8}")

    print("-" * 80)

    # 实验 6：最速下降 + Wolfe-Powell，比较 sigma2。
    for sigma2 in [0.1, 0.5, 0.9]:
        # 这里 alpha 作为 Wolfe 初始候选步长。
        _, f_opt, iters, _, _ = steepest_descent(
            x0,
            objective,
            wolfe_powell_line_search,
            alpha=sd_alpha_init,
            sigma1=1e-4,
            sigma2=sigma2,
        )
        print(f"{'SD + Wolfe':<15} & sigma2={sigma2:<11} & {f_opt:<15.2e} & {iters:<8}")

    print("=" * 80)


def main() -> None:
    """work1 实验入口。"""

    # 作业中给定的初始点与方向。
    x0 = np.array([0.0, 1.0], dtype=float)
    d0 = np.array([-1.0, -1.0], dtype=float)

    # 先做纯线搜索参数敏感性实验。
    run_line_search_experiments(x0, d0)
    # 再做“线搜索 + 最速下降”组合实验。
    run_steepest_descent_experiments(x0)


if __name__ == "__main__":
    main()
