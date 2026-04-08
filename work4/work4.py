from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimization import Objective, armijo_line_search, fr, newton_CG
from optimization.utils import ensure_dir, plot_convergence_curves, run_with_trace, save_csv, timed


BENCHMARK_PAIRS = [100, 300, 500]
EXPERIMENT_PAIRS = [1000, 10000, 30000]
GRAD_TOL = 1e-5
MAX_OUTER_ITER = 200
MAX_INNER_ITER = 100
RESULTS_DIR = ensure_dir(_PROJECT_ROOT, "work4")
PICTURE_DIR = ensure_dir(_PROJECT_ROOT, "work4", "picture")


def extended_rosenbrock_objective(n_pairs: int, optimized: bool = True) -> Objective:
    """
    扩展 Rosenbrock 函数：
        f(x) = sum_i [(1 - x_{2i-1})^2 + 10 (x_{2i} - x_{2i-1}^2)^2]
    """

    dimension = 2 * n_pairs

    if not optimized:
        # 保留逐块循环与显式 Hessian，便于和优化后版本做耗时对比。

        def func(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            total = 0.0
            for i in range(n_pairs):
                a = x[2 * i]
                b = x[2 * i + 1]
                total += (1.0 - a) ** 2 + 10.0 * (b - a**2) ** 2
            return float(total)

        def grad(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            g = np.zeros_like(x)
            for i in range(n_pairs):
                a = x[2 * i]
                b = x[2 * i + 1]
                residual = b - a**2
                g[2 * i] = -40.0 * a * residual - 2.0 * (1.0 - a)
                g[2 * i + 1] = 20.0 * residual
            return g

        def hess(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            hk = np.zeros((dimension, dimension), dtype=float)
            for i in range(n_pairs):
                a = x[2 * i]
                b = x[2 * i + 1]
                hk[2 * i, 2 * i] = 120.0 * a**2 - 40.0 * b + 2.0
                hk[2 * i, 2 * i + 1] = -40.0 * a
                hk[2 * i + 1, 2 * i] = -40.0 * a
                hk[2 * i + 1, 2 * i + 1] = 20.0
            return hk

        return Objective(func=func, grad=grad, hess=hess, name=f"extended_rosenbrock_naive_{n_pairs}")

    # 优化实现：利用切片把奇数/偶数分量分开，避免 Python 层 for 循环。
    def _split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        return x[0::2], x[1::2]

    def func(x: np.ndarray) -> float:
        odd, even = _split(x)
        residual = even - odd**2
        return float(np.sum((1.0 - odd) ** 2 + 10.0 * residual**2))

    def grad(x: np.ndarray) -> np.ndarray:
        odd, even = _split(x)
        residual = even - odd**2
        g = np.empty((dimension,), dtype=float)
        g[0::2] = -40.0 * odd * residual - 2.0 * (1.0 - odd)
        g[1::2] = 20.0 * residual
        return g

    def hess(x: np.ndarray) -> np.ndarray:
        odd, even = _split(x)
        hk = np.zeros((dimension, dimension), dtype=float)
        indices = np.arange(0, dimension, 2)
        cross = -40.0 * odd
        hk[indices, indices] = 120.0 * odd**2 - 40.0 * even + 2.0
        hk[indices, indices + 1] = cross
        hk[indices + 1, indices] = cross
        hk[indices + 1, indices + 1] = 20.0
        return hk

    def hess_vec(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        odd, even = _split(x)
        v = np.asarray(v, dtype=float)
        hv = np.empty((dimension,), dtype=float)
        diagonal = 120.0 * odd**2 - 40.0 * even + 2.0
        cross = -40.0 * odd
        # Newton-CG 只需要 Hessian 与向量的乘积，用这个接口可避免构造超大 Hessian。
        hv[0::2] = diagonal * v[0::2] + cross * v[1::2]
        hv[1::2] = cross * v[0::2] + 20.0 * v[1::2]
        return hv

    return Objective(
        func=func,
        grad=grad,
        hess=hess,
        hess_vec=hess_vec,
        name=f"extended_rosenbrock_{n_pairs}",
    )

def main() -> None:
    fr_params = {"grad_tol": GRAD_TOL, "max_outer_iter": MAX_OUTER_ITER}
    ncg_params = {
        "grad_tol": GRAD_TOL,
        "max_outer_iter": MAX_OUTER_ITER,
        "max_inner_iter": MAX_INNER_ITER,
    }

    timed_run = timed(run_with_trace)
    benchmark_rows = []
    experiment_rows = []
    convergence_histories = {}

    # 第一部分：对比 Newton-CG 在“优化前/优化后”目标函数实现上的运行时间。
    for n_pairs in BENCHMARK_PAIRS:
        x0 = np.tile(np.array([-1.2, 1.0], dtype=float), n_pairs)
        for method_name, optimized in (
            ("Newton-CG before optimization", False),
            ("Newton-CG after optimization", True),
        ):
            objective = extended_rosenbrock_objective(n_pairs, optimized=optimized)
            (result, _, _, _), runtime = timed_run(
                optimizer=newton_CG,
                x0=x0,
                objective=objective,
                line_search=armijo_line_search,
                optimizer_params=ncg_params,
            )
            benchmark_rows.append(
                {
                    "phase": "benchmark",
                    "method": method_name,
                    "n_pairs": n_pairs,
                    "dimension": 2 * n_pairs,
                    "runtime_sec": runtime,
                    "iters": result[2],
                    "f_opt": result[1],
                    "converged": result[3],
                    "grad_norm": result[4],
                }
            )

    # 第二部分：在大规模问题上比较 FR 与 Newton-CG 的效率和收敛表现。
    for n_pairs in EXPERIMENT_PAIRS:
        x0 = np.tile(np.array([-1.2, 1.0], dtype=float), n_pairs)
        objective = extended_rosenbrock_objective(n_pairs, optimized=True)
        for method_name, optimizer, optimizer_params in (
            ("FR + Armijo", fr, fr_params),
            ("Newton-CG + Armijo", newton_CG, ncg_params),
        ):
            (result, _, f_trace, _), runtime = timed_run(
                optimizer=optimizer,
                x0=x0,
                objective=objective,
                line_search=armijo_line_search,
                optimizer_params=optimizer_params,
            )
            experiment_rows.append(
                {
                    "phase": "main",
                    "method": method_name,
                    "n_pairs": n_pairs,
                    "dimension": 2 * n_pairs,
                    "runtime_sec": runtime,
                    "iters": result[2],
                    "f_opt": result[1],
                    "converged": result[3],
                    "grad_norm": result[4],
                }
            )
            if n_pairs == EXPERIMENT_PAIRS[0]:
                convergence_histories[f"{method_name} (n={n_pairs})"] = f_trace

    # 运行时间曲线：一张看程序优化收益，一张看算法间比较。
    for rows, title, filename in (
        (
            benchmark_rows,
            "Newton-CG runtime before/after program optimization",
            "work4_newton_cg_optimization_runtime.png",
        ),
        (
            experiment_rows,
            "FR vs Newton-CG on extended Rosenbrock",
            "work4_method_runtime_compare.png",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for method in sorted({row["method"] for row in rows}):
            method_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: row["n_pairs"])
            x = np.array([row["dimension"] for row in method_rows], dtype=float)
            y = np.array([row["runtime_sec"] for row in method_rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.6, label=method)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Runtime (s)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PICTURE_DIR / filename, dpi=220)
        plt.close(fig)

    if convergence_histories:
        plot_convergence_curves(
            convergence_histories,
            f"Convergence curves on extended Rosenbrock (n={EXPERIMENT_PAIRS[0]})",
            PICTURE_DIR / "work4_convergence_curve.png",
        )

    save_csv(RESULTS_DIR / "work4_results.csv", benchmark_rows + experiment_rows)

    print("Optimization benchmark (before vs after):")
    for row in benchmark_rows:
        print(
            f"  n={row['n_pairs']:>6d}, {row['method']:<30s} "
            f"time={row['runtime_sec']:.4f}s, iters={row['iters']}, "
            f"converged={row['converged']}, grad_norm={row['grad_norm']:.3e}"
        )

    print("\nMain experiment (FR vs Newton-CG):")
    for row in experiment_rows:
        print(
            f"  n={row['n_pairs']:>6d}, {row['method']:<20s} "
            f"time={row['runtime_sec']:.4f}s, iters={row['iters']}, "
            f"f={row['f_opt']:.6e}, converged={row['converged']}, grad_norm={row['grad_norm']:.3e}"
        )

    print(f"\nSaved figures to: {PICTURE_DIR}")
    print(f"Saved CSV to: {RESULTS_DIR / 'work4_results.csv'}")


if __name__ == "__main__":
    main()
