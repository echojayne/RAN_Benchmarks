# MODEL CARD: Citywide Milan iTransformer

## 1. 模型定位

这是一个用于 Milan 流量预测任务的 `citywide shared-model iTransformer`。

它不是：

- `ML-TP` 这类 per-cell meta-learning 模型
- 每个 cell 单独训练一个模型的方案

它是：

- 一个共享模型
- 在所有保留的 cell 上联合训练
- 对每个测试时刻做 rolling one-step prediction

## 2. 为什么保留这版

这版是当前整理后最稳定、最可复用的结果，原因是：

- 训练和推理代码已最小化
- 节假日异常测试窗已剔除
- 病态 `R^2 < 0` cell 已剔除
- 最终结果已收敛并保留 best checkpoint

## 3. 数据与协议

数据来源：

- Telecom Italia Milan dataset

使用方式：

- 原始 `10 min` 数据聚合到 `1h`
- 仅使用 `internet_traffic`

最终协议：

- 训练：`2013-11-01 00:00:00 UTC` 到 `2013-12-16 00:00:00 UTC`
- 测试：`2013-12-16 00:00:00 UTC` 到 `2013-12-23 00:00:00 UTC`
- 输入长度：`3`
- 预测长度：`1`
- 训练 epoch：`300`
- 保留 cell 数：`9943`
- 排除 cell 数：`57`

排除规则来源：

- 先跑无节假日版本
- 找出 `R^2 < 0` 的病态 cell
- 形成 `config/exclude_r2neg_cells.json`
- 在最终版本里训练和评估时一并排除

## 4. 指标说明

最终 summary 里保留的是 per-cell 聚合指标：

- mean/median `MAE`
- mean/median `RMSE`
- mean/median `R^2`
- mean/median `NMAE`
- mean/median `NRMSE`

需要注意：

- `global R^2` 与 `mean per-cell R^2` 不是同一个量
- citywide 文献里常见的是更接近 global/citywide 口径
- 当前归档结果文件默认保留的是 per-cell 聚合口径

## 5. 最终结果

从 `metrics/run_summary.json` 读取：

- best epoch: `285`
- best val MAE: `0.0331200238`
- mean MAE: `25.7390294742`
- median MAE: `13.3414840698`
- mean RMSE: `35.8264774336`
- median RMSE: `18.4251728058`
- mean R^2: `0.8775452250`
- median R^2: `0.9024845195`
- mean NMAE: `0.0342035585`
- median NMAE: `0.0340076052`
- mean NRMSE: `0.0472291688`
- median NRMSE: `0.0465059690`

## 6. 和文献结果的关系

这版结果更接近以下 citywide Milan 工作的协议家族：

- `STDenseNet`
- `HSTNet`
- `STCNet`
- `att-MCSTCNet`

但它不是这些方法的逐字复刻，差异主要包括：

- 使用的是 shared `iTransformer`
- 没有显式的空间卷积或图结构
- 没有 cross-domain 特征
- 测试窗显式避开节假日异常段
- 额外剔除了病态 cell

因此：

- 这版适合作为当前任务的稳定基线/可用模型
- 不适合作为对外宣称“严格复现某篇论文”的唯一依据

## 7. 已知限制

- 输入长度只有 `3`，更接近 paper-aligned `P=3` 设定，不是长上下文版本
- 没有保留中间日志与缓存，因此后续复现会重新生成缓存
- 如果要做严格 apples-to-apples 论文对比，仍需统一指标口径和协议细节

## 8. 推荐用途

推荐：

- 作为 traffic forecasting 任务的当前 `well-done` 模型包
- 作为后续 citywide 时空模型的 baseline
- 作为推理与可视化的起点

不推荐：

- 直接与 `ML-TP` 这类 per-cell adaptation 方法做“方法公平性”结论
- 在不说明指标口径的情况下，直接与外部论文表格做绝对数值比较
