from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimization import (
    Objective,
    armijo_line_search,
    bb_step_search,
    bfgc,
    fr,
    steepest_descent,
)

Result = Tuple[np.ndarray, float, int, bool, float]
Optimizer = Callable[..., Result]
LineSearch = Callable[..., Tuple[float, int]]


def _picture_dir() -> Path:
    picture_dir = _PROJECT_ROOT / "work3" / "picture"
    picture_dir.mkdir(parents=True, exist_ok=True)
    return picture_dir


def rosenbrock_objective() -> Objective:
    def func(x: np.ndarray) -> float:
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    def grad(x: np.ndarray) -> np.ndarray:
        g1 = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
        g2 = 200.0 * (x[1] - x[0] ** 2)
        return np.array([g1, g2], dtype=float)

    return Objective(func=func, grad=grad, name="rosenbrock")


def powell_singular_objective() -> Objective:
    """
    Powell 奇异函数：
    f(x)= (x1+10x2)^2 + 5(x3-x4)^2 + (x2-2x3)^4 + 10(x1-x4)^4
    """

    def func(x: np.ndarray) -> float:
        x1, x2, x3, x4 = x
        return (
            (x1 + 10.0 * x2) ** 2
            + 5.0 * (x3 - x4) ** 2
            + (x2 - 2.0 * x3) ** 4
            + 10.0 * (x1 - x4) ** 4
        )

    def grad(x: np.ndarray) -> np.ndarray:
        x1, x2, x3, x4 = x
        t1 = x1 + 10.0 * x2
        t2 = x3 - x4
        t3 = x2 - 2.0 * x3
        t4 = x1 - x4
        return np.array(
            [
                2.0 * t1 + 40.0 * (t4**3),
                20.0 * t1 + 4.0 * (t3**3),
                10.0 * t2 - 8.0 * (t3**3),
                -10.0 * t2 - 40.0 * (t4**3),
            ],
            dtype=float,
        )

    return Objective(func=func, grad=grad, name="powell_singular")


def run_with_trace(
    optimizer: Optimizer,
    x0: np.ndarray,
    objective: Objective,
    line_search: LineSearch,
    optimizer_params: Dict | None = None,
    line_search_params: Dict | None = None,
) -> Tuple[Result, List[np.ndarray], List[float], List[float]]:
    x0 = np.asarray(x0, dtype=float)
    x_trace = [x0.copy()]
    f_trace = [objective.value(x0)]
    alpha_trace: List[float] = []

    def callback(_: int, x: np.ndarray, fx: float, __: float, alpha: float) -> None:
        x_trace.append(np.asarray(x, dtype=float).copy())
        f_trace.append(float(fx))
        alpha_trace.append(float(alpha))

    kwargs: Dict = {"callback": callback}
    if optimizer_params is not None:
        kwargs.update(optimizer_params)
    if line_search_params is not None:
        kwargs.update(line_search_params)

    result = optimizer(x0, objective, line_search, **kwargs)
    return result, x_trace, f_trace, alpha_trace


def _plot_rosenbrock_bb_contour(x_trace: List[np.ndarray], save_path: Path) -> None:
    x1 = np.linspace(-2.0, 2.0, 400)
    x2 = np.linspace(-1.0, 3.0, 400)
    xx, yy = np.meshgrid(x1, x2)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    levels = np.logspace(-1, 3, 22)
    contour = ax.contour(xx, yy, zz, levels=levels, cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8)

    points = np.asarray(x_trace, dtype=float)
    ax.plot(points[:, 0], points[:, 1], "-o", color="tab:red", linewidth=1.2, markersize=2.3, label="BB path")
    ax.plot(points[0, 0], points[0, 1], "s", color="tab:blue", markersize=6, label="start")
    ax.plot(points[-1, 0], points[-1, 1], "*", color="black", markersize=9, label="end")
    ax.plot(1.0, 1.0, "x", color="tab:green", markersize=8, label=r"$x^\ast=(1,1)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Rosenbrock contour and BB trajectory")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_convergence_curves(histories: Dict[str, List[float]], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, values in histories.items():
        y = np.maximum(np.asarray(values, dtype=float), 1e-16)
        ax.semilogy(np.arange(y.size), y, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _write_summary(
    summary_path: Path,
    rosen_result: Result,
    powell_bfgc_result: Result,
    powell_fr_result: Result,
) -> None:
    lines = [
        "method,iters,f_opt,converged,grad_norm",
        (
            "Steepest+BB,"
            f"{rosen_result[2]},"
            f"{rosen_result[1]:.8e},"
            f"{rosen_result[3]},"
            f"{rosen_result[4]:.8e}"
        ),
        (
            "BFGC+Armijo(Powell),"
            f"{powell_bfgc_result[2]},"
            f"{powell_bfgc_result[1]:.8e},"
            f"{powell_bfgc_result[3]},"
            f"{powell_bfgc_result[4]:.8e}"
        ),
        (
            "FR+Armijo(Powell),"
            f"{powell_fr_result[2]},"
            f"{powell_fr_result[1]:.8e},"
            f"{powell_fr_result[3]},"
            f"{powell_fr_result[4]:.8e}"
        ),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    picture_dir = _picture_dir()

    # 实验 1：BB 步长最速下降法求解 Rosenbrock。
    rosen = rosenbrock_objective()
    x0_rosen = np.array([-1.2, 1.0], dtype=float)
    rosen_result, rosen_x_trace, rosen_f_trace, _ = run_with_trace(
        optimizer=steepest_descent,
        x0=x0_rosen,
        objective=rosen,
        line_search=bb_step_search,
        optimizer_params={"max_outer_iter": 20_000, "grad_tol": 1e-6},
        line_search_params={"alpha_init": 0.1, "variant": "auto"},
    )

    _plot_rosenbrock_bb_contour(rosen_x_trace, picture_dir / "work3_rosenbrock_bb_contour.png")
    _plot_convergence_curves(
        {"Steepest + BB": rosen_f_trace},
        "Rosenbrock convergence with BB step size",
        picture_dir / "work3_rosenbrock_bb_curve.png",
    )

    # 实验 2：Armijo 线搜索下，比较 BFGC 与 FR 在 Powell 奇异函数上的表现。
    powell = powell_singular_objective()
    x0_powell = np.array([3.0, -1.0, 0.0, 1.0], dtype=float)
    optimizer_params = {"max_outer_iter": 20_000, "grad_tol": 1e-6}
    armijo_params = {"alpha": 1.0, "sigma1": 1e-4, "rho": 0.5}

    powell_bfgc_result, _, powell_bfgc_f_trace, _ = run_with_trace(
        optimizer=bfgc,
        x0=x0_powell,
        objective=powell,
        line_search=armijo_line_search,
        optimizer_params=optimizer_params,
        line_search_params=armijo_params,
    )
    powell_fr_result, _, powell_fr_f_trace, _ = run_with_trace(
        optimizer=fr,
        x0=x0_powell,
        objective=powell,
        line_search=armijo_line_search,
        optimizer_params=optimizer_params,
        line_search_params=armijo_params,
    )

    _plot_convergence_curves(
        {
            "BFGC + Armijo": powell_bfgc_f_trace,
            "FR + Armijo": powell_fr_f_trace,
        },
        "Powell singular function convergence",
        picture_dir / "work3_powell_armijo_curve.png",
    )

    _write_summary(
        _PROJECT_ROOT / "work3" / "work3_summary.csv",
        rosen_result,
        powell_bfgc_result,
        powell_fr_result,
    )

    print("Experiment 1: Steepest + BB on Rosenbrock")
    print(
        f"  iters={rosen_result[2]}, f*={rosen_result[1]:.4e}, "
        f"converged={rosen_result[3]}, grad_norm={rosen_result[4]:.3e}"
    )
    print("Experiment 2: Armijo + {BFGC, FR} on Powell singular function")
    print(
        f"  BFGC: iters={powell_bfgc_result[2]}, f*={powell_bfgc_result[1]:.4e}, "
        f"converged={powell_bfgc_result[3]}, grad_norm={powell_bfgc_result[4]:.3e}"
    )
    print(
        f"  FR: iters={powell_fr_result[2]}, f*={powell_fr_result[1]:.4e}, "
        f"converged={powell_fr_result[3]}, grad_norm={powell_fr_result[4]:.3e}"
    )


if __name__ == "__main__":
    main()
