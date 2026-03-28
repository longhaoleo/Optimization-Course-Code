from __future__ import annotations

from typing import Tuple

import numpy as np

from ..core import Objective


def armijo_line_search(
    xk: np.ndarray,
    dk: np.ndarray,
    objective: Objective,
    alpha_init: float = 1.0,
    sigma1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 10_000,
) -> Tuple[float, int]:
    """Armijo 回溯线搜索。"""

    assert 0 < sigma1 < 0.5, "sigma1 must satisfy 0 < sigma1 < 0.5."
    assert 0 < rho < 1, "rho must satisfy 0 < rho < 1."

    alpha = alpha_init
    fxk = objective.value(xk)
    directional_derivative = float(np.dot(objective.gradient(xk), dk))
    counter = 1

    while objective.value(xk + alpha * dk) > fxk + sigma1 * alpha * directional_derivative:
        alpha *= rho
        counter += 1
        if counter >= max_iter:
            break

    return alpha, counter
