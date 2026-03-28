import numpy as np

from optimization import (
    Objective,
    armijo_line_search,
    golden_ratio_line_search,
    steepest_descent,
    wolfe_powell_line_search,
)


def quadratic_difference_objective() -> Objective:
    """work1 目标函数: f(x) = (x1 - x2)^2 + x2^2。"""

    def func(x: np.ndarray) -> float:
        return (x[0] - x[1]) ** 2 + x[1] ** 2

    def grad(x: np.ndarray) -> np.ndarray:
        df_dx1 = 2.0 * (x[0] - x[1])
        df_dx2 = -2.0 * (x[0] - x[1]) + 2.0 * x[1]
        return np.array([df_dx1, df_dx2], dtype=float)

    return Objective(func=func, grad=grad, name="quadratic_difference")


def run_line_search_experiments(x0: np.ndarray, d0: np.ndarray) -> None:
    objective = quadratic_difference_objective()
    alpha_init = 5.0

    print("=== Part 1: line-search sensitivity on fixed x0 and d0 ===")
    print("=" * 70)
    print(f"{'Method':<15} & {'Parameter':<18} & {'Alpha':<10} & {'Iters':<5}")
    print("-" * 70)

    for eps in [1e-2, 1e-4, 1e-6]:
        alpha, count = golden_ratio_line_search(x0, d0, objective, tol=eps)
        print(f"{'GoldenRatio':<15} & eps={eps:<14} & {alpha:<10.6f} & {count:<5}")

    print("-" * 70)

    rho = 0.5
    sigma1 = 1e-4
    for s1 in [1e-4, 0.1, 0.2]:
        alpha, count = armijo_line_search(
            x0,
            d0,
            objective,
            alpha_init=alpha_init,
            sigma1=s1,
            rho=rho,
        )
        print(f"{'Armijo':<15} & sigma1={s1:<11} & {alpha:<10.6f} & {count:<5}")

    for r in [0.2, 0.5, 0.8]:
        alpha, count = armijo_line_search(
            x0,
            d0,
            objective,
            alpha_init=alpha_init,
            sigma1=sigma1,
            rho=r,
        )
        print(f"{'Armijo':<15} & rho={r:<11} & {alpha:<10.6f} & {count:<5}")

    print("-" * 70)

    for s2 in [0.1, 0.5, 0.9]:
        alpha, count = wolfe_powell_line_search(
            x0,
            d0,
            objective,
            alpha=alpha_init,
            sigma1=1e-4,
            sigma2=s2,
        )
        print(f"{'Wolfe':<15} & sigma2={s2:<13} & {alpha:<10.6f} & {count:<5}")

    print("-" * 70)

    test_sigma1 = 0.2
    for beta in [0.1, 0.5, 2.0, 5.0]:
        alpha, count = wolfe_powell_line_search(
            x0,
            d0,
            objective,
            alpha=alpha_init,
            sigma1=test_sigma1,
            sigma2=0.9,
            rho=0.5,
            beta=beta,
        )
        print(f"{'Wolfe':<15} & beta={beta:<15} & {alpha:<10.6f} & {count:<5}")


def run_steepest_descent_experiments(x0: np.ndarray) -> None:
    objective = quadratic_difference_objective()
    sd_alpha_init = 1.0

    print("\n=== Part 2: steepest descent with different line searches ===")
    print("=" * 80)
    print(f"{'Method':<15} & {'Parameter':<18} & {'Opt Value f(x*)':<15} & {'SD Iters':<8}")
    print("-" * 80)

    _, f_opt, iters, _, _ = steepest_descent(
        x0,
        objective,
        golden_ratio_line_search,
        tol=1e-4,
    )
    print(f"{'SD + Golden':<15} & eps=1e-4{'':<10} & {f_opt:<15.2e} & {iters:<8}")
    print("-" * 80)

    for rho in [0.2, 0.5, 0.8]:
        _, f_opt, iters, _, _ = steepest_descent(
            x0,
            objective,
            armijo_line_search,
            alpha_init=sd_alpha_init,
            rho=rho,
            sigma1=1e-4,
        )
        print(f"{'SD + Armijo':<15} & rho={rho:<14} & {f_opt:<15.2e} & {iters:<8}")

    print("-" * 80)

    for sigma2 in [0.1, 0.5, 0.9]:
        _, f_opt, iters, _, _ = steepest_descent(
            x0,
            objective,
            wolfe_powell_line_search,
            alpha=sd_alpha_init,
            sigma1=1e-4,
            sigma2=sigma2,
        )
        print(f"{'SD + Wolfe':<15} & sigma2={sigma2:<11} & {f_opt:<15.2e} & {iters:<8}")

    print("=" * 80)


def main() -> None:
    x0 = np.array([0.0, 1.0], dtype=float)
    d0 = np.array([-1.0, -1.0], dtype=float)

    run_line_search_experiments(x0, d0)
    run_steepest_descent_experiments(x0)


if __name__ == "__main__":
    main()
