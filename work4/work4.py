from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimization import (
    armijo_line_search,
    conjugate_gradient_fr,
    conjugate_gradient_inexact_newton,
    extended_rosenbrock_objective,
)
from optimization.utils import ensure_dir, plot_convergence_curves, run_with_trace, save_csv, timed


BENCHMARK_PAIRS = [100, 300, 500]
EXPERIMENT_PAIRS = [1000, 10000, 30000]
INITIAL_PAIR = np.array([-1.2, 1.0], dtype=float)
GRAD_TOL = 1e-5
MAX_OUTER_ITER = 200
MAX_INNER_ITER = 100
RESULTS_DIR = ensure_dir(_PROJECT_ROOT, "work4")
PICTURE_DIR = ensure_dir(_PROJECT_ROOT, "work4", "picture")
RESULTS_PATH = RESULTS_DIR / "work4_results.csv"


def main() -> None:
    fr_params = {"grad_tol": GRAD_TOL, "max_outer_iter": MAX_OUTER_ITER}
    inexact_newton_params = {
        "grad_tol": GRAD_TOL,
        "max_outer_iter": MAX_OUTER_ITER,
        "max_inner_iter": MAX_INNER_ITER,
    }

    timed_run = timed(run_with_trace)
    benchmark_rows = []
    experiment_rows = []
    convergence_histories = {}

    # 第一部分：对比 Newton-CG 在优化前/优化后目标函数实现上的运行时间。
    for n_pairs in BENCHMARK_PAIRS:
        x0 = np.tile(INITIAL_PAIR, n_pairs)
        for method_name, optimized in (
            ("Newton-CG before optimization", False),
            ("Newton-CG after optimization", True),
        ):
            objective = extended_rosenbrock_objective(n_pairs, optimized=optimized)
            (result, _, _, _), runtime = timed_run(
                optimizer=conjugate_gradient_inexact_newton,
                x0=x0,
                objective=objective,
                line_search=armijo_line_search,
                optimizer_params=inexact_newton_params,
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
        x0 = np.tile(INITIAL_PAIR, n_pairs)
        objective = extended_rosenbrock_objective(n_pairs, optimized=True)
        for method_name, optimizer, optimizer_params in (
            ("FR + Armijo", conjugate_gradient_fr, fr_params),
            ("Newton-CG + Armijo", conjugate_gradient_inexact_newton, inexact_newton_params),
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
            "wnewton_cg_optimization_runtime.png",
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

    save_csv(RESULTS_PATH, benchmark_rows + experiment_rows)

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
    print(f"Saved CSV to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
