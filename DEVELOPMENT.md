# GMD-SGT Release Note

## Version

`v0.1`

## 概述

`v0.1` 是 GMD-SGT 的首个发布版本。本版本的目标不是覆盖全部高级功能，而是完成一条清晰、可维护、可继续扩展的主链路：

- 数据读取
- 模型训练
- checkpoint 保存与恢复
- 推理接口
- TorchScript 导出
- 基础测试覆盖

## 本版本已完成内容

### 模型

- 完成统一的等变 MLIP 主体结构
- 支持局域消息传递与长程模块组合
- 支持 `none`、`invariant_attention`、`equivariant_attention`、`electrostatic`
- 支持 PBC 邻居图构建
- 修正 batched PBC neighbor graph 的分图构建与回拼逻辑

### 数据

- 支持 `extxyz` 数据读取
- 支持 `npz` 数据读取
- 支持 train/val/test 划分
- 支持按元素统计参考能量偏移

### 训练

- 完成 `EnergyForceLoss`
- 完成 `Trainer`
- 支持 checkpoint 保存
- 支持 resume 恢复
- 支持 early stopping
- 支持 CSV 日志记录

### 推理与导出

- 完成 `MLIPCalculator`
- 支持 Python 侧推理
- 支持 ASE calculator 接口
- 支持 TorchScript 导出

### 测试

- 等变性测试
- 推理接口测试
- 导出流程测试
- batched PBC 回归测试

## 项目特色

- 以等变建模为核心，强调物理对称性
- 训练、推理、导出链路完整
- 代码结构模块化，便于替换模型子模块
- 支持与外部 MD 程序进一步集成
- 为后续版本迭代保留了清晰扩展点

## 如何使用

### 安装

```bash
pip install -e .
```

开发环境：

```bash
pip install -e .[dev]
```

### 训练

```bash
python scripts/train_cli.py --config configs/default.yaml
```

或：

```bash
python scripts/train_cli.py --config configs/water.yaml
```

恢复训练：

```bash
python scripts/train_cli.py --config configs/default.yaml --resume outputs/run/ckpt_best.pt
```

### Smoke Test

```bash
python scripts/smoke_test.py
```

### 导出模型

```bash
python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt --output model.pt
```

### Python 推理

```python
from gmd_sgt.inference import MLIPCalculator

calc = MLIPCalculator.from_checkpoint("outputs/run/ckpt_best.pt", device="cpu")
```

## v0.1 定位

`v0.1` 是一个首发版本，重点是把主流程打通，而不是把所有性能与工程细节都做到最终形态。后续版本可以继续增强：

- 更完整的训练验证覆盖
- 更强的部署与导出兼容性
- 更多数据格式支持
- 更成熟的模型结构与性能优化

## 结论

从版本定位上看，`v0.1` 已经具备发布说明、基础使用说明和后续继续演进的基础。
