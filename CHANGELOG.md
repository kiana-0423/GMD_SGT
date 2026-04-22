# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- 新增轻量 `AllegroStyleBackbone`，支持 `species`、`positions`、`cell`、`neighbor_list` 输入，采用 Bessel radial basis 与方向基函数构建 local energy-only backbone，forces 统一由 `-grad(E, positions)` 得到
- 新增 Stage 1 训练入口 `gmd_sgt.training.train_backbone` 与配置文件 `configs/stage1_backbone.yaml`，复用现有 dataset、loss、trainer、checkpoint 机制，并支持 `--dry-run` 最小数据集验证
- 新增 `GMDSGTModel` staged residual-learning 总装模型，支持 `E = E_backbone + lambda_gnn * DeltaE_GNN + lambda_attn * DeltaE_Attn` 的保守能量组合形式
- 新增 `GNNCorrection` 残差分支，仅输出 residual atomic energy `DeltaE_GNN`
- 新增 `TransformerCorrection` 稀疏 / 局域 attention 残差分支接口，作为可选 correction branch，默认在 Stage 2 配置中关闭
- 新增 Stage 2 训练入口 `gmd_sgt.training.train_residual` 与配置文件 `configs/stage2_residual.yaml`，支持加载 backbone checkpoint，并提供 `freeze_backbone` 与 `semi_freeze_backbone` 两种训练模式
- 新增模型工厂与 checkpoint 兼容层，统一支持 `UnifiedEquivariantMLIP`、`AllegroStyleBackbone`、`GMDSGTModel` 的实例化和恢复
- 新增 staged MLIP 相关最小测试，覆盖 model forward、energy-force consistency、backbone freeze 行为与 staged checkpoint round-trip
- 新增结构数据校验模块，对 `energy`、`forces`、`cell/pbc`、原子种类映射及可选标签做基础一致性检查

### Changed

- `train()` 稳定 API 现在会根据 `model.type` 自动分流到 legacy 单阶段训练、Stage 1 backbone 训练或 Stage 2 residual 训练，尽量保持现有调用方式不变
- 推理 calculator 与 TorchScript 导出路径改为按 checkpoint 中的 `model_type` 自动加载模型，使 staged 模型可以沿用现有推理 / 导出接口
- 训练 checkpoint 现在额外记录 `model_type`，`Trainer.from_checkpoint()` 可以自动恢复正确的模型类
- Stage 1 backbone forward 现在会暴露 residual 分支所需的 invariant node features、distance、radial basis、coordination 与 neighbor graph 中间量
- README 补充了 Stage 1 backbone 与 Stage 2 residual 的最小运行方式和 dry-run 示例

### Fixed

- 修复 batch 中仅部分样本带有 `cell` 或 `stress` 时可能发生的静默字段丢失问题，改为在 `collate_fn` 中显式报错
- 数据读取流程在 `extxyz` 与 `npz` 路径上统一接入结构校验，避免缺失标签或形状不一致的数据样本悄然进入训练流程

## v0.2

### Added

- 新增稳定的 Python API 层，提供 `train()`、`export_model()`、`predict()` 与 `OnlinePredictor`，便于外部 adapter 以程序化方式接入
- 新增面向在线监督的结构化推理结果 `PredictionResult`，稳定返回 `energy`、`forces`、`ensemble_forces`、`latent_descriptor`、`unsafe_probability`、`metadata`
- 新增配置化的 `online_monitoring` 配置段，支持按开关控制能量、ensemble force、latent descriptor、unsafe probability 等输出
- 新增基于多 checkpoint committee 的 ensemble 推理能力，可返回 `ensemble_forces`，shape 为 `(M, N, 3)`
- 新增输入结构校验与基础异常处理，支持 `positions`、`species` 或 `symbols`、`cell`、`pbc`、`edge_index`、`edge_shift`
- 新增在线监督接口测试，覆盖单模型推理、ensemble 推理、非法输入、训练接口与导出接口

### Changed

- 训练 CLI 复用新的稳定训练 API，同时保持原有 subprocess/命令行使用方式兼容
- 导出 CLI 复用新的稳定导出 API，明确返回导出产物路径
- README 新增在线监督接口、配置项说明、ensemble 开启方式与返回 schema 文档

### Fixed

- 修复模型在位置梯度为空时可能缺失 `forces` 输出的问题，改为稳定返回零力张量
- 改进 TorchScript 导出流程，在脚本化失败时回退到 tracing，以提升导出兼容性

### Reserved

- `latent_descriptor` 已预留接口，当前若模型未暴露中间表征则返回 `None`
- `unsafe_probability` 已预留接口，当前若模型未提供风险头则返回 `None`

## v0.1

- 首个可用版本，完成模型训练、基础推理、TorchScript 导出、数据读取与基础测试链路
