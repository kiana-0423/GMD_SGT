# GMD-SE3GNN v0.1

GMD-SE3GNN 是一个面向原子体系建模的 SE(3)/E(3) 等变机器学习势能项目，目标是同时支持训练、推理、TorchScript 导出，以及与外部分子动力学程序的集成。

## Release Note

当前版本：`v0.1`

`v0.1` 是项目的首个可用版本，重点完成了以下能力：

- 统一的等变势能模型结构，支持局域消息传递与长程模块组合
- 数据读取与训练管线，支持 `extxyz` 和 `npz` 数据格式
- 训练组件模块化，包括 loss、trainer、resume、checkpoint、CSV 日志
- 推理接口与 TorchScript 导出，便于对接外部程序
- PBC 邻居图构建与 batched PBC 路径修正
- 基础测试覆盖，包括等变性、推理接口与导出流程

## 项目特色

- 等变建模：围绕原子坐标与几何关系设计，强调旋转等变、平移不变与置换不变
- 模块化结构：模型、训练、数据、推理分目录组织，便于继续扩展
- 长程机制可选：支持 `none`、`invariant_attention`、`equivariant_attention`、`electrostatic`
- 训练与部署衔接清晰：训练完成后可以直接导出为 TorchScript
- 对接外部模拟程序友好：支持外部传入 `edge_index` 和 `edge_shift`

## 目录结构

```text
gmd_se3gnn/
  data/         # 数据读取、切分、统计、dataset
  inference/    # 推理接口、TorchScript 导出
  models/       # 模型主体、消息传递、长程模块、PBC 邻居图
  training/     # loss、trainer
  __init__.py
  model.py      # 兼容导出入口
  train.py      # 兼容训练入口

configs/
  default.yaml
  water.yaml

scripts/
  train_cli.py
  smoke_test.py
  export_model.py

tests/
```

## 安装

建议先创建虚拟环境，再安装依赖。

```bash
pip install -e .
```

如果需要开发与测试依赖：

```bash
pip install -e .[dev]
```

如果你是按源码手动安装，常用依赖包括：

```bash
pip install torch ase e3nn torch_cluster torch_scatter pyyaml tqdm pytest
```

## 如何使用

### 1. 准备数据

目前支持两类输入：

- `extxyz`
- `npz`

在配置文件中指定训练数据路径，例如：

```yaml
data:
  train_file: your_dataset.extxyz
```

### 2. 开始训练

使用默认配置：

```bash
python scripts/train_cli.py --config configs/default.yaml
```

使用水体系示例配置：

```bash
python scripts/train_cli.py --config configs/water.yaml
```

从 checkpoint 恢复训练：

```bash
python scripts/train_cli.py --config configs/default.yaml --resume outputs/run/ckpt_best.pt
```

### 3. 运行 smoke test

```bash
python scripts/smoke_test.py
```

### 4. 导出 TorchScript 模型

```bash
python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt --output model.pt
```

### 5. Python 推理

```python
from gmd_se3gnn.inference import MLIPCalculator
import numpy as np

calc = MLIPCalculator.from_checkpoint("outputs/run/ckpt_best.pt", device="cpu")

positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
species = np.array([6, 1], dtype=np.int64)

result = calc.compute(positions=positions, species=species)
print(result["energy"])
print(result["forces"])
```

### 6. ASE 接口

```python
atoms.calc = calc.get_ase_calculator()
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## 配置建议

- `configs/default.yaml`：通用默认训练配置
- `configs/water.yaml`：水体系示例配置

如果是第一次跑，建议先从较小模型和较小 batch 开始，先确认数据、训练和导出链路都能跑通。

## 当前版本说明

`v0.1` 更适合作为：

- 原型验证版本
- 模型训练与导出流程打通版本
- 后续扩展更高精度实现与工程优化的基础版本

## License

本项目采用仓库内 `LICENSE` 文件所声明的许可证。
