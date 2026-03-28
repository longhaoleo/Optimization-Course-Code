from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

Array = np.ndarray
ObjectiveFunction = Callable[[Array], float]
GradientFunction = Callable[[Array], Array]
HessianFunction = Callable[[Array], Array]


@dataclass(frozen=True)
class Objective:
    """目标函数与梯度的统一封装。"""

    func: ObjectiveFunction
    grad: GradientFunction
    name: str = "objective"
    hess: Optional[HessianFunction] = None

    def value(self, x: Array) -> float:
        """计算函数值。"""
        x = np.asarray(x, dtype=float)
        return float(self.func(x))

    def gradient(self, x: Array) -> Array:
        """计算梯度向量。"""
        x = np.asarray(x, dtype=float)
        return np.asarray(self.grad(x), dtype=float)

    def hessian(self, x: Array) -> Array:
        """计算 Hessian 矩阵。"""
        if self.hess is None:
            raise ValueError("Hessian is not provided for this objective.")
        x = np.asarray(x, dtype=float)
        return np.asarray(self.hess(x), dtype=float)
