from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ..core import Objective

LineSearch = Callable[..., Tuple[float, int]]
IterationCallback = Callable[[int, np.ndarray, float, float, float], None]


def _inexact_cg_direction(
    matvec: Callable[[np.ndarray], np.ndarray],
    gk: np.ndarray,
    max_inner_iter: int,
    negative_curvature_tol: float = 1e-12,
) -> Tuple[np.ndarray, int]:
    """
    用课件中的 CG 内循环近似求解 A d = -g。

    返回：
        dk: 近似牛顿方向
        inner_iters: 内层 CG 迭代次数
    """

    grad_norm = float(np.linalg.norm(gk))
    eps_k = min(0.5, np.sqrt(grad_norm)) * grad_norm

    dk = np.zeros_like(gk)
    rk = gk.copy()
    pk = -rk

    for inner_iter in range(max_inner_iter):
        apk = matvec(pk)
        curvature = float(np.dot(pk, apk))
        if curvature <= negative_curvature_tol:
            # 负曲率时按算法 7.1 回退。
            if inner_iter == 0:
                return -gk.copy(), inner_iter + 1
            return dk, inner_iter + 1

        rr = float(np.dot(rk, rk))
        alpha = rr / curvature
        dk_next = dk + alpha * pk
        rk_next = rk + alpha * apk

        if float(np.linalg.norm(rk_next)) <= eps_k:
            return dk_next, inner_iter + 1

        beta = float(np.dot(rk_next, rk_next) / max(rr, 1e-32))
        pk = -rk_next + beta * pk
        dk = dk_next
        rk = rk_next

    return dk, max_inner_iter


def newton_CG(
    x0: np.ndarray,
    objective: Objective,
    line_search_func: LineSearch,
    grad_tol: float = 1e-5,
    max_outer_iter: int = 200,
    max_inner_iter: int | None = None,
    callback: Optional[IterationCallback] = None,
    **ls_params,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """
    Line-search Newton-CG 方法。

    若 objective 提供 hess_vec，则优先使用 Hessian-向量积；
    否则回退为显式 Hessian 与向量相乘。
    """

    if objective.hess is None and objective.hess_vec is None:
        raise ValueError("Newton-CG requires objective.hess or objective.hess_vec.")

    xk = np.asarray(x0, dtype=float)
    converged = False
    x_prev: Optional[np.ndarray] = None
    g_prev: Optional[np.ndarray] = None

    for iteration in range(max_outer_iter):
        gk = objective.gradient(xk)
        grad_norm = float(np.linalg.norm(gk))
        if grad_norm < grad_tol:
            converged = True
            break

        inner_limit = max_inner_iter if max_inner_iter is not None else min(xk.size, 200)
        if objective.hess_vec is not None:
            matvec = lambda v: objective.hessian_vector_product(xk, v)
        else:
            hk = objective.hessian(xk)
            matvec = lambda v: hk @ v

        dk, _ = _inexact_cg_direction(matvec, gk, inner_limit)
        if float(np.dot(gk, dk)) >= 0.0:
            dk = -gk

        alpha, _ = line_search_func(
            xk,
            dk,
            objective,
            x_prev=x_prev,
            g_prev=g_prev,
            gk=gk,
            iteration=iteration,
            **ls_params,
        )

        x_old = xk.copy()
        xk = xk + alpha * dk
        x_prev = x_old
        g_prev = gk.copy()

        if callback is not None:
            callback(
                iteration + 1,
                xk.copy(),
                objective.value(xk),
                grad_norm,
                float(alpha),
            )
    else:
        iteration = max_outer_iter
        grad_norm = float(np.linalg.norm(objective.gradient(xk)))

    return (
        xk,
        objective.value(xk),
        iteration,
        converged,
        grad_norm,
    )
