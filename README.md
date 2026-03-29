# Optimization Course Practice

课程代码按“可复用模块 + 作业调用脚本”组织：

- `optimization/`：算法模块（可复用）
- `work1/`、`work2/`：作业脚本与作业文件（调用模块）

## 当前结构

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
│     ├─ steepest_descent.py
│     ├─ newton.py
│     └─ modified_newton.py
├─ work1/
│  ├─ __init__.py
│  └─ work1.py
└─ work2/
   ├─ __init__.py
   ├─ work2.py
   ├─ newton.py          # 兼容入口，转调 work2.py
   └─ a9a_train.csv
```

## 模块说明

- `optimization/core.py`
  - `Objective`：统一封装 `func/grad/hess`

- `optimization/line_search/`
  - `golden_ratio_line_search`
  - `armijo_line_search`
  - `wolfe_powell_line_search`

- `optimization/optimizers/`
  - `steepest_descent`
  - `newton_method`
  - `modified_newton_method`
  - 返回统一为：
    `(x_opt, f_opt, iters, converged, grad_norm)`

## 运行方式

### work1

```bash
python -m work1.work1
```

### work2（生成实验素材）

```bash
python -m work2.work2
```

说明：不使用命令行参数，是否运行 Rastrigin / a9a 逻辑回归由 `work2/work2.py` 顶部开关控制：
`RUN_RASTRIGIN`、`RUN_LOGISTIC`、`A9A_PATH`。
