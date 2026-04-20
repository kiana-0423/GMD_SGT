# Changelog

All notable changes to this project will be documented in this file.

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
