from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np

from .core import Objective

T = TypeVar("T")


def ensure_dir(path: Path, *parts: str) -> Path:
    """创建目录并返回对应 Path。"""

    target = path.joinpath(*parts)
    target.mkdir(parents=True, exist_ok=True)
    return target


def timed(func: Callable[..., T]) -> Callable[..., Tuple[T, float]]:
    """返回一个包装函数，执行原函数并统计耗时。"""

    def wrapper(*args: object, **kwargs: object) -> Tuple[T, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return wrapper


def run_with_trace(
    optimizer,
    x0: np.ndarray,
    objective: Objective,
    line_search,
    optimizer_params: Dict | None = None,
    line_search_params: Dict | None = None,
    store_x_trace: bool = False,
) -> tuple[tuple[np.ndarray, float, int, bool, float], List[np.ndarray], List[float], List[float]]:
    """运行优化器并记录迭代轨迹。"""

    x0 = np.asarray(x0, dtype=float)
    x_trace = [x0.copy()] if store_x_trace else []
    f_trace = [objective.value(x0)]
    alpha_trace: List[float] = []

    def callback(_: int, x: np.ndarray, fx: float, __: float, ___: float) -> None:
        if store_x_trace:
            x_trace.append(np.asarray(x, dtype=float).copy())
        f_trace.append(float(fx))
        alpha_trace.append(float(___))

    kwargs = {"callback": callback}
    if optimizer_params is not None:
        kwargs.update(optimizer_params)
    if line_search_params is not None:
        kwargs.update(line_search_params)

    result = optimizer(x0, objective, line_search, **kwargs)
    return result, x_trace, f_trace, alpha_trace


def save_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    """把字典列表写成 CSV。"""

    rows = list(rows)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_convergence_curves(
    histories: Dict[str, List[float]],
    title: str,
    save_path: Path,
    ylabel: str = "Objective value",
) -> None:
    """绘制收敛曲线。"""

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, values in histories.items():
        x = np.arange(len(values), dtype=float)
        y = np.maximum(np.asarray(values, dtype=float), 1e-16)
        ax.semilogy(x, y, marker="o", linewidth=1.6, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
