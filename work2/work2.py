
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# 兼容两种运行方式：
# 1) 在项目根目录执行：python -m work2.work2
# 2) 直接运行文件：python work2/work2.py
# 第二种情况下，sys.path[0] 会变成 work2 目录，导致无法 import optimization，
# 这里项目根目录加入 sys.path。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optimization import (
    Objective,
    armijo_line_search,
    golden_ratio_line_search,
    modified_newton_method,
    newton_method,
    steepest_descent,
    wolfe_powell_line_search,
)

# 统一结果格式：
# (x_opt, f_opt, iters, converged, grad_norm)
Result = Tuple[np.ndarray, float, int, bool, float]
Optimizer = Callable[..., Result]
# LineSearch 约定：输入 (xk, dk, objective, **params)，输出 (alpha, line_search_iters)
LineSearch = Callable[..., Tuple[float, int]]


def _picture_dir() -> Path:
    """图片统一输出到 work2/picture。"""

    picture_dir = _PROJECT_ROOT / "work2" / "picture"
    picture_dir.mkdir(parents=True, exist_ok=True)
    return picture_dir


def rosenbrock_objective() -> Objective:
    """
    Rosenbrock 测试函数（二维）：
        f(x) = 100*(x2 - x1^2)^2 + (1 - x1)^2
    特点：谷底狭长弯曲，常用于测试优化算法在病态曲面上的表现。
    """

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


def rastrigin_objective(n: int = 6) -> Objective:
    """
    Rastrigin 函数（n 维）：
        f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    特点：非凸、多峰，容易陷入局部极小值，适合比较算法鲁棒性。
    """

    def func(x: np.ndarray) -> float:
        return 10.0 * n + float(np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    def hess(x: np.ndarray) -> np.ndarray:
        diag = 2.0 + 40.0 * (np.pi**2) * np.cos(2.0 * np.pi * x)
        return np.diag(diag)

    return Objective(func=func, grad=grad, hess=hess, name=f"rastrigin_{n}d")


def logistic_objective_from_csv(csv_path: Path) -> Objective:
    """
    a9a 逻辑回归目标函数：
        (1/m) * sum(log(1 + exp(-b_i * a_i^T x))) + lambda||x||^2
        lambda = 1 / (100m)
    CSV 约定：第一列为标签 b_i，其余列为特征 a_i。

    与作业 PDF 的记号对应关系：
        - 代码里的 y    <-> 公式里的 b
        - 代码里的 xmat <-> 公式里的 A
        - 代码里的 w    <-> 公式里的 x
    因此 y * (xmat @ w) 就是在批量计算每个样本的 b_i a_i^T x。

    返回：
        Objective(func, grad, hess)
    """

    # 读入数据并拆成标签/特征矩阵。
    # 这里默认标签已经编码成 {-1, 1}，与作业里的 b_i 写法一致。
    data = np.loadtxt(csv_path, delimiter=",", dtype=float)
    y = data[:, 0]
    xmat = data[:, 1:]
    m, n = xmat.shape
    # 与作业一致的 L2 正则系数，m 越大正则越弱。
    lam = 1.0 / (100.0 * m)
    eye = np.eye(n, dtype=float)

    def func(w: np.ndarray) -> float:
        # yz[i] = b_i * a_i^T x
        yz = y * (xmat @ w)
        # logaddexp(0, -yz) 等价于 log(1 + exp(-b_i a_i^T x))，
        return float(np.logaddexp(0.0, -yz).mean() + lam * np.dot(w, w))

    def grad(w: np.ndarray) -> np.ndarray:
        # yz = b_i * a_i^T x，后面公式更紧凑。
        yz = y * (xmat @ w)
        # prob[i] = 1 / (1 + exp(b_i a_i^T x))
        # 也就是 sigmoid(-b_i a_i^T x)。
        prob = 1.0 / (1.0 + np.exp(yz))
        # 梯度公式：
        #   -(1/m) * sum(b_i a_i * sigmoid(-b_i a_i^T x)) + 2 * lambda * x
        # 写成矩阵形式后就是下面这一行。
        return -(xmat.T @ (y * prob)) / m + 2.0 * lam * w

    def hess(w: np.ndarray) -> np.ndarray:
        yz = y * (xmat @ w)
        prob = 1.0 / (1.0 + np.exp(yz))
        # weight[i] = sigmoid(-b_i a_i^T x) * (1 - sigmoid(-b_i a_i^T x))
        # 对应 Hessian 中对角权重矩阵 W 的第 i 个对角元。
        weight = prob * (1.0 - prob)
        # Hessian = A^T W A / m + 2 * lambda * I
        # 这里没有显式构造巨大的对角矩阵 W，而是利用广播把每一行样本
        # 按 weight[i] 缩放后，再做矩阵乘法，结果完全等价。
        return (xmat.T * weight) @ xmat / m + 2.0 * lam * eye

    return Objective(func=func, grad=grad, hess=hess, name="logistic_a9a")


def plot_rosenbrock_landscape() -> None:
    """
    绘制 Rosenbrock 函数等高线图。
    这张图主要用于在报告里展示 Rosenbrock 狭长弯曲谷底的形状。
    """

    x1 = np.linspace(-2.0, 2.0, 400)
    x2 = np.linspace(-1.0, 3.0, 400)
    xx, yy = np.meshgrid(x1, x2)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2

    fig, ax = plt.subplots(figsize=(6, 4.5))
    levels = np.logspace(-1, 3, 20)
    contour = ax.contour(xx, yy, zz, levels=levels, cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8)

    # 在图上标出两组作业中使用的典型初始点，以及理论极小点 (1,1)。
    ax.plot(1.2, 1.2, "ro", label=r"$x_0=(1.2,1.2)$")
    ax.plot(-1.2, 1.0, "bs", label=r"$x_0=(-1.2,1.0)$")
    ax.plot(1.0, 1.0, "k*", markersize=10, label=r"$x^\ast=(1,1)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Rosenbrock contour")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_picture_dir() / "rosenbrock_contour.png", dpi=200)
    plt.close(fig)


def plot_rastrigin_landscape() -> None:
    """
    绘制二维 Rastrigin 函数热力图。
    实验本身使用 n=6，但二维图更直观，适合放在报告里说明多峰结构。
    """

    x1 = np.linspace(-4.5, 4.5, 500)
    x2 = np.linspace(-4.5, 4.5, 500)
    xx, yy = np.meshgrid(x1, x2)
    zz = 20.0 + xx**2 + yy**2 - 10.0 * np.cos(2.0 * np.pi * xx) - 10.0 * np.cos(2.0 * np.pi * yy)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    image = ax.contourf(xx, yy, zz, levels=60, cmap="plasma")
    fig.colorbar(image, ax=ax, label="f(x)")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Rastrigin contour (2D view)")
    ax.grid(True, linestyle="--", alpha=0.15)
    fig.tight_layout()
    fig.savefig(_picture_dir() / "rastrigin_contour.png", dpi=200)
    plt.close(fig)


def run_with_trace(
    optimizer: Optimizer,
    x0: np.ndarray,
    objective: Objective,
    line_search: LineSearch,
    **kwargs,
) -> Tuple[Result, List[float], List[float]]:
    """
    通用执行器：
    1. 运行优化算法
    2. 记录每轮目标函数值
    3. 记录每轮线搜索步长 alpha

    参数：
        optimizer: 优化算法（最速下降/牛顿/修正牛顿）
        x0: 初始点
        objective: 目标函数对象
        line_search: 线搜索函数
        **kwargs: 统一透传给 optimizer

    返回：
        result: 优化结果五元组
        values: 每轮目标函数值（包含初始值）
        alphas: 每轮线搜索得到的步长
    """

    # values[0] 对应初始点 x0 的函数值，便于和后续迭代曲线对齐。
    values = [objective.value(x0)]
    alphas: List[float] = []

    # 回调函数由优化器在每轮迭代结束后调用。
    # 这里只关心 fx 和 alpha，其他参数用占位符变量忽略。
    def callback(_: int, __: np.ndarray, fx: float, ___: float, alpha: float) -> None:
        values.append(float(fx))
        alphas.append(float(alpha))

    result = optimizer(
        x0,
        objective,
        line_search,
        callback=callback,
        **kwargs,
    )
    return result, values, alphas



def run_rosenbrock_alpha_trace() -> None:
    """
    作业第 2/3 部分：
    - 初始点 x0=(1.2, 1.2)
    - 比较最速下降法、牛顿法、修正牛顿法
    - 打印每次迭代的函数值 f(x_k)
    - 绘制目标函数值-迭代次数曲线
    """

    objective = rosenbrock_objective()
    x0 = np.array([1.2, 1.2], dtype=float)

    methods: Dict[str, Optimizer] = {
        "Steepest": steepest_descent,
        "Newton": newton_method,
        "ModifiedNewton": modified_newton_method,
    }

    histories: Dict[str, List[float]] = {}
    print("\n=== Part A: Rosenbrock from x0=(1.2, 1.2) ===")
    for name, method in methods.items():
        result, values, _ = run_with_trace(
            method,
            x0,
            objective,
            wolfe_powell_line_search,
            grad_tol=1e-5,
            max_outer_iter=20_000 ,
            alpha=1.0,
            sigma1=1e-4,
            sigma2=0.9,
            rho=0.5,
            beta=0.5,
        )
        histories[name] = values

        print(
            f"{name:<15} f*={result[1]:.4e}, iters={result[2]}, "
            f"converged={result[3]}, grad_norm={result[4]:.2e}"
        )
        print(f"{name:<15} f(x) trace:")
        for k, fx in enumerate(values, start=0):
            if name == "Steepest": # 收敛较慢，1000次迭代打印一次
                if k % 1000 == 0: print(f"  iter={k:05d}, f={float(fx):.6e}")
            else: print(f"  iter={k:05d}, f={float(fx):.6e}")

    # 绘制收敛曲线
    fig, ax = plt.subplots()
    # 半对数坐标, 方便观察收敛曲线变化规律。
    for name, values in histories.items():
        ax.semilogy(values, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title("Rosenbrock objective vs iteration")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(_picture_dir() / "rosenbrock_curve.png", dpi=200)
    plt.close(fig)


def run_line_search_comparison() -> None:
    """
    作业第 4 部分：
    - 初始点 x0=(-1.2, 1.0)
    - 比较三种线搜索（Golden/Armijo/Wolfe）
    - 在三种下降法上统计迭代次数与收敛结果
    """

    objective = rosenbrock_objective()
    x0 = np.array([-1.2, 1.0], dtype=float)

    # 下降方向算法：最速下降法、牛顿法、修正牛顿法
    methods: Dict[str, Optimizer] = {
        "Steepest": steepest_descent,
        "Newton": newton_method,
        "ModifiedNewton": modified_newton_method,
    }

    line_searches: Dict[str, Tuple[LineSearch, dict]] = {
        "GoldenRatio": (golden_ratio_line_search, {"ak": 0.0, "bk": 2.0, "tol": 1e-4}),
        "Armijo": (armijo_line_search, {"alpha_init": 1.0, "sigma1": 1e-4, "rho": 0.5}),
        "WolfePowell": (
            wolfe_powell_line_search,
            {"alpha": 1.0, "beta": 0.5, "sigma1": 1e-4, "sigma2": 0.9, "rho": 0.5},
        ),
    }

    print("\n=== Part B: x0=(-1.2,1.0), line-search comparison ===")
    print(f"{'Method':<16} {'LineSearch':<14} {'Iters':<8} {'f*':<14} {'Converged':<10}")
    print("-" * 70)

    for m_name, method in methods.items():
        for ls_name, (ls_func, ls_params) in line_searches.items():
            result = method(
                x0,
                objective,
                ls_func,
                grad_tol=1e-5,
                max_outer_iter=20_000,
                **ls_params,
            )
            print(
                f"{m_name:<16} {ls_name:<14} {result[2]:<8d} "
                f"{result[1]:<14.4e} {str(result[3]):<10}"
            )


def run_rastrigin_experiment() -> None:
    """
    作业第 5 部分：
    选取多个初始点，在 Rastrigin 函数上比较三种下降法。
    """

    objective = rastrigin_objective(n=6)
    # 多组初始点用于观察非凸多峰函数下的初值敏感性。
    initials = [
        np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=float),
        np.array([-3.0, -3.0, -3.0, -3.0, -3.0, -3.0], dtype=float),
        np.array([2.5, -1.5, 3.0, -2.0, 1.0, -3.5], dtype=float),
    ]

    methods: Dict[str, Optimizer] = {
        "Steepest": steepest_descent,
        "Newton": newton_method,
        "ModifiedNewton": modified_newton_method,
    }

    print("\n=== Part C: Rastrigin (n=6) ===")
    for idx, x0 in enumerate(initials, start=1):
        print(f"\n{idx}. Initial point: {x0}")
        for name, method in methods.items():
            result = method(
                x0,
                objective,
                wolfe_powell_line_search,
                grad_tol=1e-5,
                max_outer_iter=20_000,
                alpha=1.0,
                beta=0.5,
                sigma1=1e-4,
                sigma2=0.9,
                rho=0.5,
            )
            print(
                f"{name:<15} f*={result[1]:.4e}, iters={result[2]}, "
                f"converged={result[3]}"
            )


def run_logistic_a9a(csv_path: Path) -> None:
    """
    作业第 6 部分：
    使用 a9a 数据集构建逻辑回归目标，并比较三种下降法。
    """

    objective = logistic_objective_from_csv(csv_path)
    x0 = np.zeros(123, dtype=float)

    # 一阶法给更大迭代预算，二阶法通常更快收敛。
    methods: Dict[str, Tuple[Optimizer, dict]] = {
        "Steepest": (steepest_descent, {"max_outer_iter": 2000, "grad_tol": 1e-5}),
        "Newton": (newton_method, {"max_outer_iter": 80, "grad_tol": 1e-6}),
        "ModifiedNewton": (modified_newton_method, {"max_outer_iter": 80, "grad_tol": 1e-6}),
    }

    print("\n=== Part D: a9a logistic regression ===")
    print(f"{'Method':<16} {'Iters':<8} {'f*':<14} {'Converged':<10} {'GradNorm':<12}")
    print("-" * 70)
    histories: Dict[str, List[float]] = {}
    for name, (method, cfg) in methods.items():
        result, values, _ = run_with_trace(
            method,
            x0,
            objective,
            wolfe_powell_line_search,
            alpha=1.0,
            beta=0.5,
            sigma1=1e-4,
            sigma2=0.9,
            rho=0.5,
            **cfg,
        )
        histories[name] = values

        print(
            f"{name:<16} {result[2]:<8d} {result[1]:<14.4e} "
            f"{str(result[3]):<10} {result[4]:<12.3e}"
        )

    # 画出逻辑回归目标函数随迭代次数变化（半对数）。
    fig, ax = plt.subplots()
    for name, values in histories.items():
        ax.semilogy(values, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title("a9a logistic regression: objective vs iteration")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(_picture_dir() / "logistic_curve.png", dpi=200)
    plt.close(fig)


def main() -> None:
    plot_rosenbrock_landscape()
    plot_rastrigin_landscape()
    run_rosenbrock_alpha_trace()
    run_line_search_comparison()
    run_rastrigin_experiment()
    run_logistic_a9a(_PROJECT_ROOT / "work2" / "a9a_train.csv")


if __name__ == "__main__":
    main()
