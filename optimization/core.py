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
        # 强制转 float ndarray，避免输入 list/int 引起类型不一致。
        x = np.asarray(x, dtype=float)
        # 统一转成 Python float，便于打印和序列化。
        return float(self.func(x))

    def gradient(self, x: Array) -> Array:
        """计算梯度向量。"""
        # 与 value 同样做输入标准化。
        x = np.asarray(x, dtype=float)
        # 输出也统一成 float ndarray。
        return np.asarray(self.grad(x), dtype=float)

    def hessian(self, x: Array) -> Array:
        """计算 Hessian 矩阵。"""
        # 若未提供 Hessian，显式抛错，提醒调用者改用一阶方法或补齐接口。
        if self.hess is None:
            raise ValueError("Hessian is not provided for this objective.")
        # 保持与 value/gradient 一致的输入输出规范。
        x = np.asarray(x, dtype=float)
        return np.asarray(self.hess(x), dtype=float)
