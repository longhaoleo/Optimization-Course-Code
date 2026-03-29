from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ..core import Objective

LineSearch = Callable[..., Tuple[float, int]]
IterationCallback = Callable[[int, np.ndarray, float, float, float], None]


def steepest_descent(
    x0: np.ndarray,
    objective: Objective,
    line_search_func: LineSearch,
    grad_tol: float = 1e-5,
    max_outer_iter: int = 1000,
    callback: Optional[IterationCallback] = None,
    **ls_params,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """
    最速下降法。

    返回值：
        (x_opt, f_opt, iters, converged, grad_norm)
    """

    # 把初始点统一转成 float ndarray，避免整数数组导致步长更新精度问题。
    xk = np.asarray(x0, dtype=float)
    # 是否满足停止条件的标志位。
    converged = False

    # 外层主循环：每次迭代更新一次 xk。
    for iteration in range(max_outer_iter):
        # 1) 计算当前梯度 gk。
        gk = objective.gradient(xk)
        # 2) 用梯度二范数衡量一阶最优性。
        grad_norm = float(np.linalg.norm(gk))
        # 3) 若梯度足够小，则判定收敛并停止。
        if grad_norm < grad_tol:
            converged = True
            break

        # 4) 最速下降方向：dk = -gk。
        dk = -gk
        # 5) 线搜索决定本轮步长 alpha。
        alpha, _ = line_search_func(xk, dk, objective, **ls_params)
        # 6) 位置更新：x_{k+1} = x_k + alpha * d_k。
        xk = xk + alpha * dk

        # 7) 若用户提供回调钩子，则把本轮信息回传给上层调用。
        if callback is not None:
            callback(
                # 迭代编号使用 1-based，更符合报告展示习惯。
                iteration + 1,
                # copy() 防止外部误改内部状态。
                xk.copy(),
                # 当前目标函数值。
                objective.value(xk),
                # 当前梯度范数。
                grad_norm,
                # 当前线搜索步长。
                alpha,
            )
    else:
        # for-else: 仅当循环未break时进入，表示达到最大迭代次数。
        iteration = max_outer_iter
        # 退出前再计算一次梯度范数，保证返回值与最终 xk 对齐。
        grad_norm = float(np.linalg.norm(objective.gradient(xk)))

    # 返回统一五元组，便于与牛顿法/修正牛顿法对齐比较。
    return (
        xk,
        objective.value(xk),
        iteration,
        converged,
        grad_norm,
    )
