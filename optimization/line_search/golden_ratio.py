from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core import Objective


def golden_ratio_line_search(
    xk: np.ndarray,
    dk: np.ndarray,
    objective: Objective,
    ak: float = 0.0,
    bk: float = 2.0,
    tol: float = 1e-5,
    max_iter: int = 10_000,
) -> Tuple[float, int]:
    """黄金分割精确线搜索。"""

    assert ak < bk, "ak must satisfy ak < bk."
    assert tol > 0, "tol must be positive."

    phi = lambda alpha: objective.value(xk + alpha * dk)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0

    lam = bk - ratio * (bk - ak)
    mu = ak + ratio * (bk - ak)
    phi_lam = phi(lam)
    phi_mu = phi(mu)
    counter = 1

    while abs(bk - ak) >= tol and counter < max_iter:
        if phi_lam < phi_mu:
            bk = mu
            mu = lam
            phi_mu = phi_lam
            lam = bk - ratio * (bk - ak)
            phi_lam = phi(lam)
        else:
            ak = lam
            lam = mu
            phi_lam = phi_mu
            mu = ak + ratio * (bk - ak)
            phi_mu = phi(mu)
        counter += 1

    return (ak + bk) / 2.0, counter
