# Traffic Forecasting iTransformer Well-Done Package

该目录存放 Milan 流量预测任务当前整理后的 `iTransformer` 可复用模型包。

目标不是保留完整实验历史，而是保留一套最小但完整的交付物：

- 模型定义
- 训练入口
- 推理入口
- 最终配置
- 最终权重
- 最终评估结果

## 目录结构

- `code/itransformer_citywide.py`
  - 模型定义
  - 数据读取与小时级聚合
  - 训练/验证/测试逻辑
- `code/run_traffic_forecasting_itransformer_citywide.py`
  - 训练入口
- `code/infer_traffic_forecasting_itransformer_citywide.py`
  - 推理入口
- `config/itransformer_citywide.yaml`
  - 当前最终训练与推理配置
- `config/exclude_r2neg_cells.json`
  - 需要排除的病态 cell 列表
- `checkpoints/best.pt`
  - 最终保留权重
- `metrics/run_summary.json`
  - 聚合结果
- `metrics/protocol_manifest.json`
  - 实际训练/测试协议与切分信息
- `metrics/per_cell_metrics.csv`
  - 每个 cell 的评估结果
- `MODEL_CARD.md`
  - 模型说明、协议、结果和解读

## 当前模型对应的协议

- 数据集：Telecom Italia Milan
- 任务：citywide traffic forecasting
- 聚合粒度：`1h`
- 目标变量：`internet_traffic`
- 训练时间窗：`2013-11-01 00:00:00 UTC` 到 `2013-12-16 00:00:00 UTC`
- 测试时间窗：`2013-12-16 00:00:00 UTC` 到 `2013-12-23 00:00:00 UTC`
- 说明：节假日异常周未纳入测试窗
- 输入长度：`3`
- 预测方式：rolling one-step prediction
- 训练 epoch：`300`
- 排除 cell：`57`
- 保留 cell：`9943`

## 最终结果

来自 `metrics/run_summary.json`：

- best epoch: `285`
- best val MAE: `0.0331200238`
- mean MAE: `25.7390`
- median MAE: `13.3415`
- mean RMSE: `35.8265`
- median RMSE: `18.4252`
- mean R^2: `0.8775`
- median R^2: `0.9025`
- mean NMAE: `0.03420`
- median NMAE: `0.03401`
- mean NRMSE: `0.04723`
- median NRMSE: `0.04651`

## 训练

在仓库根目录执行：

```bash
python tasks/traffic_prediction/methods/itransformer/citywide_compat/code/run_traffic_forecasting_itransformer_citywide.py \
  --config tasks/traffic_prediction/methods/itransformer/citywide_compat/config/itransformer_citywide.yaml
```

说明：

- 代码会自动从原始 Milan 数据构建小时级矩阵缓存
- 默认使用配置文件里给出的输出目录
- 如果需要复现实验，建议先检查配置中的路径是否与当前环境一致

## 推理

在仓库根目录执行：

```bash
python tasks/traffic_prediction/methods/itransformer/citywide_compat/code/infer_traffic_forecasting_itransformer_citywide.py \
  --config tasks/traffic_prediction/methods/itransformer/citywide_compat/config/itransformer_citywide.yaml \
  --checkpoint tasks/traffic_prediction/methods/itransformer/weights/well_done_best.pt
```

默认会导出：

- `predictions_test.npz`

其中包含：

- `predictions`
- `targets`
- `cell_ids`
- `timestamps`

## 使用建议

- 如果做论文式对比，优先参考 `metrics/protocol_manifest.json`，不要只看配置文件名
- 如果关心整体城市预测效果，参考 global/citywide 聚合指标
- 如果关心单 cell 泛化，参考 `metrics/per_cell_metrics.csv`
- `R^2 < 0` 的病态 cell 已经从最终保留版本中剔除

## 备注

该目录是整理后的 `well-done` 归档版本，不包含：

- 中间日志
- 训练缓存
- 旧实验分支
- 诊断绘图
- 临时测试脚本
