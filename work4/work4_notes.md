# Work4 说明

默认初始点取重复的 `(-1.2, 1.0)`，对应扩展 Rosenbrock 的常见测试方式。

程序优化点：
1. 把目标函数、梯度从 Python for 循环改成 NumPy 向量化实现。
2. 为 Newton-CG 增加 Hessian-向量积接口，避免在大规模实验里显式构造完整 Hessian。
3. trace 统一由公共 `run_with_trace` 记录，脚本本身只保留实验流程。

优化前后 benchmark：
- n=100 (dim=200), Newton-CG before optimization: time=0.0071s, iters=19, converged=True, grad_norm=1.793e-08
- n=100 (dim=200), Newton-CG after optimization: time=0.0013s, iters=19, converged=True, grad_norm=1.793e-08

主实验结果：
- n=1000 (dim=2000), FR + Armijo: time=0.0052s, iters=50, converged=False, f=9.381222e-08, grad_norm=1.314e-03
- n=1000 (dim=2000), Newton-CG + Armijo: time=0.0017s, iters=19, converged=True, f=2.119141e-17, grad_norm=6.092e-08
