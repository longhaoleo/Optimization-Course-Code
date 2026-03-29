# Optimization Course Practice

最优化课程代码按**可复用模块——optimizition** 和 **作业脚本_work\***组织：

- `optimization/`：算法模块
- `work1/`、`work2/`：各次作业的调用代码、数据与报告文件

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
python work1/work1.py
```

### work2

```bash
python work2/work2.py
```

运行 `work2` 后会：

- 在终端输出各实验结果
- 在 `work2/picture/` 下生成函数图和收敛曲线
- 供 `work2/work2.tex` 直接引用生成实验报告
