from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core import Objective


def bb_step_search(
    xk: np.ndarray,
    _dk: np.ndarray,
    objective: Objective,
    x_prev: np.ndarray | None = None,
    alpha_init: float = 0.1,
    variant: str = "auto",
    eps: float = 1e-12,
    **_: object,
) -> Tuple[float, int]:
    """
    Barzilai-Borwein (BB) 步长搜索。

    该方法通过相邻两点的信息估计步长，核心量为
        s_k = x_k - x_{k-1},
        y_k = grad f(x_k) - grad f(x_{k-1}).

    为了兼容当前优化器统一调用接口，这里保留了第二个位置参数 `_dk`；
    BB 公式实际使用的是 `xk`、`x_prev` 和 `objective`。

    参数：
        variant:
            - "bb1": alpha = (s_k^T s_k) / (s_k^T y_k)
            - "bb2": alpha = (s_k^T y_k) / (y_k^T y_k)
            - "auto": 在正值候选里取更保守的一个
    """

    if x_prev is None:
        return float(alpha_init), 0

    xk = np.asarray(xk, dtype=float)
    x_prev = np.asarray(x_prev, dtype=float)

    # 计算 s_k 和 y_k。
    sk = xk - x_prev
    yk = objective.gradient(xk) - objective.gradient(x_prev)

    sTy = float(np.dot(sk, yk))
    sTs = float(np.dot(sk, sk))
    yTy = float(np.dot(yk, yk))

    # 数值检查，防止溢出
    alpha_bb1 = None
    alpha_bb2 = None
    if abs(sTy) > eps:
        alpha_bb1 = sTs / sTy
    if yTy > eps:
        alpha_bb2 = sTy / yTy

    variant_key = variant.strip().lower()
    if variant_key == "bb1":
        alpha = alpha_bb1
    elif variant_key == "bb2":
        alpha = alpha_bb2
    elif variant_key == "auto":
        candidates = [a for a in (alpha_bb1, alpha_bb2) if a is not None and a > 0.0]
        alpha = min(candidates) if candidates else None
    else:
        raise ValueError("variant must be one of {'bb1', 'bb2', 'auto'}.")

    if alpha is None or (not np.isfinite(alpha)) or alpha <= 0.0:
        alpha = alpha_init

    return float(alpha), 0

