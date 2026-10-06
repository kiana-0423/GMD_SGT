# Changelog

All notable changes to this project will be documented in this file.

## v0.3

包版本 `0.3.0`（`gmd_sgt.__version__`，`pyproject.toml` 通过 `tool.setuptools.dynamic` 读取）。

### Fixed

- 安装：构建后端改为 `setuptools.build_meta`；`gmd-train` 入口改为包内的 `gmd_sgt.cli:main`，wheel 安装后可在源码目录之外使用（`scripts/train_cli.py` 保留为薄封装）
- Stage 1 / Stage 2 训练：力的 `autograd.grad` 在训练时 `retain_graph` 与 `create_graph` 一致，能量 + 力损失的 `loss.backward()` 不再因计算图被释放而失败；推理时计算图立即释放
- 等变特征初始化：`UnifiedEquivariantMLIP` 只把物种嵌入写入 `0e` 通道，`l > 0` 通道初始为零，由与球谐的张量积生成；irreps 必须以 `scalar_dim` 个 `0e` 开头，且所有声明的 irreps 必须可由 `l_max`、`n_blocks` 生成，否则报错
- `l_max`：`EquivariantLongRangeBlock` 的球谐 irreps 由 `l_max` 推导（新增 `l_max` 参数，默认 2），与模型计算的边球谐一致；`l_max` 非法时报错
- 无 e3nn：非标量 irreps 的 `UnifiedEquivariantMLIP` 构造时抛出 `ImportError`；纯标量 irreps 使用依赖距离的不变连续滤波消息传递（不再与几何无关），由纯 PyTorch 模块精确复现 e3nn 的标量张量积 / Linear / BatchNorm，标量 checkpoint 可在有无 e3nn 的环境间互相加载。backbone 的闭式 l=2 基函数采用与 e3nn 相同的归一化（与 e3nn 数值一致），`l_max > 2` 无 e3nn 时报错
- PBC：使用 ASE `primitive_neighbor_list`；按轴 PBC 在数据读取、collate（`batch["pbc"]`，形状 `[n_graphs, 3]`）、Trainer、`MLIPCalculator`、ASE calculator、`OnlinePredictor` 与邻居图构建中保留；单个布尔值仍然兼容；`torch_cluster.radius_graph` 不再截断邻居数
- 截断连续性：backbone、统一模型消息传递、GNN 与 Transformer 残差分支的消息都乘以平滑截断包络；Transformer 稀疏 attention 使用包络加权并带零 logit 的“空”键归一化，边离开截断或最后一个邻居消失时能量与力连续
- TorchScript 导出：模型新增可脚本化的 `compute_energy()` 核心，导出使用 `torch.jit.script`（不再回退到 tracing），e3nn BatchNorm 在导出时冻结为等价的逐分量仿射变换；导出后重新加载并与 eager 的能量和力在多种原子数 / 边数（含空边列表与周期性平移）上比较，不一致时抛出 `ExportValidationError`
- 空验证集：`split_dataset` 校验划分比例；Trainer 在验证集为空时改用训练损失选择 checkpoint 与 early stopping，并在 checkpoint 中记录 `selection_metric`
- stress：实现 `stress = (1/V) dE/dε`（三类模型、`MLIPCalculator(compute_stress=True)`、ASE `get_stress()`）；`w_stress > 0` 时缺少 stress 标签或周期晶胞会直接报错；npz 的 `virial` 按 `-virial / V` 转换为 stress
- 力的 RMSE 损失在误差恰为零时梯度为 NaN 的问题
- 截断包络数值稳定性：`PolynomialCutoff` 改用与原多项式完全等价的因式分解形式 `(1-u)^3 · Σ_{k<p} C(k+2,2) u^k`（`1-u` 由 `(r_c-r)/r_c` 计算），float32 下靠近截断时不再出现负值（例如 r=2.99365 处原为 -5.7e-6），值与一、二阶导数在截断处仍为零
- 包络加权 attention：稳定化最大值改为基于 `s_e + log w_e` 且排除零权重边，去掉改变概率的分母 clamp；零权重边与删除该边完全等价，大 logit、极小权重下数值正确，一阶 / 二阶导数有限
- 恢复训练：`Trainer.from_checkpoint()` 以 checkpoint 中的 `model_config` 为准（模型由它与权重重建），显式传入的 `model_config` 只做一致性检查，除 `atomic_energies`（由 checkpoint buffer 决定）外任何差异都会在训练前报错；保存的 checkpoint 始终记录实际模型的配置。显式 `model_cls` 与 checkpoint 的 `model_type` 不一致时报错。从 best checkpoint 恢复到新的输出目录且之后没有改进时，新目录中仍会写入该 best checkpoint
- early stopping：先更新计数器再保存 checkpoint，best / 周期 checkpoint 保存的都是对应 epoch 结束后的状态，恢复后与不中断的训练在同一 epoch 停止
- TorchScript 导出的 `electrostatic` 模型在 `edge_shift` 非零时报错（与 eager 一致）；四张量接口无法表达所有周期性情况，限制见 README
- 静电模块的距离计算在对角元处的梯度安全性；静电模块输入周期性结构时报错（不做 Ewald 求和）

### Changed

- 长程 attention 明确为结构内全局 attention；`lr_cutoff` 仅作信息用途，未使用的 `radial_basis_lr` / `cutoff_env_lr` 模块被移除（旧 checkpoint 中的对应键在加载时忽略）；未知的 `long_range_type` 会报错
- checkpoint 新增 `format_version` 与 `selection_metric`；加载旧格式 checkpoint 会警告预测结果已因上述修正而改变
- `get_model_class()` 对未知 `model_type` 报错而不是静默回退
- `EnergyForceLoss` 的 `w_stress` 默认值由 `0.01` 改为 `0.0`：旧默认值从未生效（stress 未实现），现在 `w_stress > 0` 表示显式请求 stress 监督

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
