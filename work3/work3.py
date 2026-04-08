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
    bfgs,
    fr,
    steepest_descent,
)
from optimization.utils import ensure_dir, plot_convergence_curves, run_with_trace, save_csv

Result = Tuple[np.ndarray, float, int, bool, float]
Optimizer = Callable[..., Result]
LineSearch = Callable[..., Tuple[float, int]]
PICTURE_DIR = ensure_dir(_PROJECT_ROOT, "work3", "picture")

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


def _write_summary(
    summary_path: Path,
    rosen_result: Result,
    powell_bfgs_result: Result,
    powell_fr_result: Result,
) -> None:
    rows = [
        {
            "method": "Steepest+BB",
            "iters": rosen_result[2],
            "f_opt": f"{rosen_result[1]:.8e}",
            "converged": rosen_result[3],
            "grad_norm": f"{rosen_result[4]:.8e}",
        },
        {
            "method": "bfgs+Armijo(Powell)",
            "iters": powell_bfgs_result[2],
            "f_opt": f"{powell_bfgs_result[1]:.8e}",
            "converged": powell_bfgs_result[3],
            "grad_norm": f"{powell_bfgs_result[4]:.8e}",
        },
        {
            "method": "FR+Armijo(Powell)",
            "iters": powell_fr_result[2],
            "f_opt": f"{powell_fr_result[1]:.8e}",
            "converged": powell_fr_result[3],
            "grad_norm": f"{powell_fr_result[4]:.8e}",
        },
    ]
    save_csv(summary_path, rows)


def main() -> None:
    # 实验 1：BB 步长最速下降法求解 Rosenbrock。
    rosen = rosenbrock_objective()
    x0_rosen = np.array([-1.2, 1.0], dtype=float)
    rosen_result, rosen_x_trace, rosen_f_trace, _ = run_with_trace(
        optimizer=steepest_descent,
        x0=x0_rosen,
        objective=rosen,
        line_search=bb_step_search,
        optimizer_params={"max_outer_iter": 20_000, "grad_tol": 1e-6},
        store_x_trace=True,
    )

    _plot_rosenbrock_bb_contour(rosen_x_trace, PICTURE_DIR / "work3_rosenbrock_bb_contour.png")
    plot_convergence_curves(
        {"Steepest + BB": rosen_f_trace},
        "Rosenbrock convergence with BB step size",
        PICTURE_DIR / "work3_rosenbrock_bb_curve.png",
    )

    # 实验 2：Armijo 线搜索下，比较 bfgs 与 FR 在 Powell 奇异函数上的表现。
    powell = powell_singular_objective()
    x0_powell = np.array([3.0, -1.0, 0.0, 1.0], dtype=float)
    optimizer_params = {"max_outer_iter": 20_000, "grad_tol": 1e-6}

    powell_bfgs_result, _, powell_bfgs_f_trace, _ = run_with_trace(
        optimizer=bfgs,
        x0=x0_powell,
        objective=powell,
        line_search=armijo_line_search,
        optimizer_params=optimizer_params,
    )
    powell_fr_result, _, powell_fr_f_trace, _ = run_with_trace(
        optimizer=fr,
        x0=x0_powell,
        objective=powell,
        line_search=armijo_line_search,
        optimizer_params=optimizer_params,
    )

    plot_convergence_curves(
        {
            "bfgs + Armijo": powell_bfgs_f_trace,
            "FR + Armijo": powell_fr_f_trace,
        },
        "Powell singular function convergence",
        PICTURE_DIR / "work3_powell_armijo_curve.png",
    )

    _write_summary(
        _PROJECT_ROOT / "work3" / "work3_summary.csv",
        rosen_result,
        powell_bfgs_result,
        powell_fr_result,
    )

    print("Experiment 1: Steepest + BB on Rosenbrock")
    print(
        f"iters={rosen_result[2]}, f*={rosen_result[1]:.4e}, "
        f"converged={rosen_result[3]}, grad_norm={rosen_result[4]:.3e}"
    )
    print("Experiment 2: Armijo + {bfgs, FR} on Powell singular function")
    print(
        f"bfgs: iters={powell_bfgs_result[2]}, f*={powell_bfgs_result[1]:.4e}, "
        f"converged={powell_bfgs_result[3]}, grad_norm={powell_bfgs_result[4]:.3e}"
    )
    print(
        f"FR: iters={powell_fr_result[2]}, f*={powell_fr_result[1]:.4e}, "
        f"converged={powell_fr_result[3]}, grad_norm={powell_fr_result[4]:.3e}"
    )


if __name__ == "__main__":
    main()
