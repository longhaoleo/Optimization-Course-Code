# Optimization Course Practice

这个仓库用于最优化课程代码实践，目标是：

- 核心算法放在 `optimization` 下，便于复用
- 每次作业在 `workX` 下写调用和实验，不重复实现算法

## 目录结构

```text
.
├─ optimization/
│  ├─ __init__.py
│  ├─ core.py
│  ├─ line_search/
│  │  ├─ __init__.py
│  │  ├─ armijo.py
│  │  ├─ golden_ratio.py
│  │  └─ wolfe_powell.py
│  └─ optimizers/
│     ├─ __init__.py
│     └─ steepest_descent.py
├─ work1/
│  ├─ __init__.py
│  └─ work1.py
└─ README.md
```

## 模块说明

- `optimization/core.py`  
  定义 `Objective`，统一封装目标函数、梯度和可选 Hessian。

- `optimization/line_search/`  
  线搜索方法，每个方法一个文件：
  - `golden_ratio_line_search`
  - `armijo_line_search`
  - `wolfe_powell_line_search`

- `optimization/optimizers/steepest_descent.py`  
  最速下降法实现，返回 tuple：
  `(x_opt, f_opt, iters, converged, grad_norm)`

- `work1/work1.py`  
  作业入口脚本，定义本次目标函数并调用 `optimization` 中的方法做实验。

## 运行方式

在仓库根目录执行：

```bash
python -m work1.work1
```
