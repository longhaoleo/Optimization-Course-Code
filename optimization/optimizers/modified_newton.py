from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from ..core import Objective

LineSearch = Callable[..., Tuple[float, int]]
IterationCallback = Callable[[int, np.ndarray, float, float, float], None]


def _modified_cholesky(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    修正 Cholesky 分解（课件 Algorithm 5），用于把 Hessian 修成“可分解且正定”的形式。

    目标：给定对称矩阵 A，构造正定矩阵 B，并给出分解：
        B = L D L^T
    其中 L 为单位下三角（对角为 1），D = diag(d1,...,dn) 且 di > 0。

    关键修正点（保证正定）：在第 j 步，先算
        theta_j = max_{i>j} |c_ij|
    再选
        d_j = max(|c_jj|, (theta_j/beta)^2, delta)
    并用
        c_ii <- c_ii - c_ij^2 / d_j
    更新后续对角，从而保证所有 d_j 都有正的下界。

    参数（按课件符号）：
    - epsilon：机器精度（float64 下约为 2.22e-16）
    - gamma(A) = max_i |a_ii|
    - xi(A) = max_{i!=j} |a_ij|
    - delta = epsilon * max(gamma(A) + xi(A), 1)
    - beta = sqrt(max(gamma(A), xi(A)/sqrt(n^2-1), epsilon))

    返回：
    - L：单位下三角矩阵（对角为 1）
    - d：对角向量（D = diag(d)，且 d_i > 0）
    """

    a = np.asarray(a, dtype=float)
    assert a.ndim == 2 and a.shape[0] == a.shape[1], "A must be a square matrix."

    n = a.shape[0]
    if n == 0:
        return np.eye(0, dtype=float), np.zeros((0,), dtype=float)

    eps = float(np.finfo(float).eps)  # 机器精度 epsilon
    gamma = float(np.max(np.abs(np.diag(a))))  # gamma(A) = max_i |a_ii|
    if n == 1:
        xi = 0.0
    else:
        off = a - np.diag(np.diag(a))
        xi = float(np.max(np.abs(off)))  # xi(A) = max_{i!=j} |a_ij|

    # delta = epsilon * max(gamma(A) + xi(A), 1)，用于给 d_j 一个正的下界。
    delta = eps * max(gamma + xi, 1.0)
    # beta = sqrt(max(gamma(A), xi(A)/sqrt(n^2-1), epsilon))，控制非对角项的影响。
    denom = float(np.sqrt(n * n - 1.0)) if n > 1 else 1.0
    beta = float(np.sqrt(max(gamma, xi / denom, eps)))

    c = a.copy()
    l = np.eye(n, dtype=float)
    d = np.zeros((n,), dtype=float)

    for j in range(n):
        # 下面的 j 对应课件里的第 (j+1) 步（课件是 1-based，这里是 0-based）。

        # 计算 L 的第 j 行：l_js = c_js / d_s,  s < j
        for s in range(j):
            l[j, s] = c[j, s] / d[s]

        # 计算消元后的列元素：c_ij = a_ij - sum_{s<j} l_js * c_is,  i > j
        for i in range(j + 1, n):
            if j == 0:
                c[i, j] = a[i, j]
            else:
                c[i, j] = a[i, j] - float(np.dot(l[j, :j], c[i, :j]))

        # theta_j = max_{i>j} |c_ij|，刻画当前列非对角元素的最大幅度。
        theta = float(np.max(np.abs(c[j + 1 :, j]))) if j < n - 1 else 0.0
        # d_j = max(|c_jj|, (theta_j/beta)^2, delta)，保证 d_j >= delta > 0。
        d[j] = max(abs(float(c[j, j])), (theta / beta) ** 2, delta)

        # 更新后续对角：c_ii <- c_ii - c_ij^2/d_j，相当于做一次 LDL^T 消元。
        for i in range(j + 1, n):
            c[i, i] = c[i, i] - (c[i, j] ** 2) / d[j]

    return l, d


def _solve_from_ldlt(l: np.ndarray, d: np.ndarray, b: np.ndarray) -> np.ndarray:
    """解线性方程 (L D L^T) x = b，其中 L 为单位下三角，D=diag(d)。"""

    n = l.shape[0]
    b = np.asarray(b, dtype=float)
    assert l.shape == (n, n), "L must be square."
    assert d.shape == (n,), "d must have shape (n,)."
    assert b.shape == (n,), "b must have shape (n,)."

    # 前代：L y = b
    y = b.copy()
    for i in range(n):
        if i > 0:
            y[i] = y[i] - float(np.dot(l[i, :i], y[:i]))

    # 对角解：D z = y
    z = y / d

    # 回代：L^T x = z
    x = z.copy()
    for i in range(n - 1, -1, -1):
        if i < n - 1:
            x[i] = x[i] - float(np.dot(l[i + 1 :, i], x[i + 1 :]))
    return x


def modified_newton(
    x0: np.ndarray,
    objective: Objective,
    line_search_func: LineSearch,
    grad_tol: float = 1e-5,
    max_outer_iter: int = 200,
    callback: Optional[IterationCallback] = None,
    **ls_params,
) -> Tuple[np.ndarray, float, int, bool, float]:
    """
    修正牛顿法。

    当 Hessian 非正定时，使用“修正 Cholesky 分解（LDL^T）”构造正定近似矩阵并解方向。
    返回值：
        (x_opt, f_opt, iters, converged, grad_norm)
    """

    # 初始点统一类型。
    xk = np.asarray(x0, dtype=float)
    # 收敛标记。
    converged = False
    # 记录上一步信息，兼容需要历史量的步长策略（如 BB）。
    x_prev: Optional[np.ndarray] = None
    g_prev: Optional[np.ndarray] = None

    # 外层主循环。
    for iteration in range(max_outer_iter):
        # 1) 当前梯度与收敛判据。
        gk = objective.gradient(xk)
        grad_norm = float(np.linalg.norm(gk))
        if grad_norm < grad_tol:
            converged = True
            break

        # 2) 用修正 Cholesky 得到 Bk = L D L^T（正定），再解 Bk * dk = -gk。
        hk = objective.hessian(xk)
        l, d = _modified_cholesky(hk)
        dk = _solve_from_ldlt(l, d, -gk)

        # 3) 保险策略：若不是下降方向，退化为最速下降方向。
        if float(np.dot(gk, dk)) >= 0.0:
            dk = -gk

        # 4) 线搜索统一由外部注入，和最速下降/牛顿法保持同一接口。
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
        # 5) 更新迭代点。
        xk = xk + alpha * dk
        x_prev = x_old
        g_prev = gk.copy()

        # 6) 向外暴露迭代过程。
        if callback is not None:
            callback(
                iteration + 1,
                xk.copy(),
                objective.value(xk),
                grad_norm,
                alpha,
            )
    else:
        # 未提前收敛时，记录最终梯度范数。
        iteration = max_outer_iter
        grad_norm = float(np.linalg.norm(objective.gradient(xk)))

    # 返回统一五元组。
    return (
        xk,
        objective.value(xk),
        iteration,
        converged,
        grad_norm,
    )
