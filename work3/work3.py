import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimization import (
    Objective,
    armijo_line_search,
    bb_step_search,
    bfgc,
    cg,
    steepest_descent,
    wolfe_powell_line_search,
)

Result = Tuple[np.ndarray, float, int, bool, float]
Optimizer = Callable[..., Result]
LineSearch = Callable[..., Tuple[float, int]]


def rosenbrock_objective() -> Objective:
    """work3 默认测试函数：Rosenbrock。"""

    def func(x: np.ndarray) -> float:
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    def grad(x: np.ndarray) -> np.ndarray:
        g1 = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
        g2 = 200.0 * (x[1] - x[0] ** 2)
        return np.array([g1, g2], dtype=float)

    def hess(x: np.ndarray) -> np.ndarray:
        h11 = 1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0
        h12 = -400.0 * x[0]
        h22 = 200.0
        return np.array([[h11, h12], [h12, h22]], dtype=float)

    return Objective(func=func, grad=grad, hess=hess, name="rosenbrock")


def run_single_case(
    x0: np.ndarray,
    objective: Objective,
    optimizer: Optimizer,
    line_search: LineSearch,
    optimizer_params: Dict | None = None,
    line_search_params: Dict | None = None,
) -> Result:
    """统一入口：直接传入优化器和步长搜索函数。"""

    kwargs: Dict = {}
    if optimizer_params is not None:
        kwargs.update(optimizer_params)
    if line_search_params is not None:
        kwargs.update(line_search_params)

    return optimizer(
        np.asarray(x0, dtype=float),
        objective,
        line_search,
        **kwargs,
    )


def run_batch(
    x0: np.ndarray,
    objective: Objective,
    experiments: Iterable[Dict],
) -> None:
    """批量执行配置化实验。"""

    print(f"{'Optimizer':<16} {'LineSearch':<14} {'Iters':<8} {'f*':<14} {'Converged':<10}")
    print("-" * 72)
    for item in experiments:
        result = run_single_case(
            x0=x0,
            objective=objective,
            optimizer=item["optimizer"],
            line_search=item["line_search"],
            optimizer_params=item.get("optimizer_params"),
            line_search_params=item.get("line_search_params"),
        )
        print(
            f"{item['optimizer_name']:<16} {item['line_search_name']:<14} "
            f"{result[2]:<8d} {result[1]:<14.4e} {str(result[3]):<10}"
        )


def main() -> None:
    objective = rosenbrock_objective()
    x0 = np.array([-1.2, 1.0], dtype=float)

    # 这里就是 work3 的“可配置实验区”：后续加方法只需要加一项配置。
    experiments = [
        {
            "optimizer_name": "Steepest",
            "optimizer": steepest_descent,
            "line_search_name": "Armijo",
            "line_search": armijo_line_search,
            "optimizer_params": {"max_outer_iter": 20_000, "grad_tol": 1e-6},
            "line_search_params": {"alpha_init": 1.0, "sigma1": 1e-4, "rho": 0.5},
        },
        {
            "optimizer_name": "CG",
            "optimizer": cg,
            "line_search_name": "WolfePowell",
            "line_search": wolfe_powell_line_search,
            "optimizer_params": {"max_outer_iter": 20_000, "grad_tol": 1e-6},
            "line_search_params": {
                "alpha": 1.0,
                "beta": 0.5,
                "sigma1": 1e-4,
                "sigma2": 0.9,
                "rho": 0.5,
            },
        },
        {
            "optimizer_name": "BFGC",
            "optimizer": bfgc,
            "line_search_name": "WolfePowell",
            "line_search": wolfe_powell_line_search,
            "optimizer_params": {"max_outer_iter": 20_000, "grad_tol": 1e-6},
            "line_search_params": {
                "alpha": 1.0,
                "beta": 0.5,
                "sigma1": 1e-4,
                "sigma2": 0.9,
                "rho": 0.5,
            },
        },
        {
            "optimizer_name": "Steepest",
            "optimizer": steepest_descent,
            "line_search_name": "BB",
            "line_search": bb_step_search,
            "optimizer_params": {"max_outer_iter": 20_000, "grad_tol": 1e-6},
            "line_search_params": {"alpha_init": 0.1},
        },
    ]

    run_batch(x0=x0, objective=objective, experiments=experiments)


if __name__ == "__main__":
    main()
