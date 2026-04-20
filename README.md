# GMD-SE3GNN v0.2

GMD-SE3GNN 是一个面向原子体系建模的 SE(3)/E(3) 等变机器学习势能项目，目标是同时支持训练、推理、TorchScript 导出，以及与外部分子动力学程序的集成。

## Release Note

当前版本：`v0.2`

`v0.2` 在 `v0.1` 的基础上，补齐了面向外部在线监督框架接入所需的稳定接口，重点包括：

- 稳定的训练、导出、在线推理 Python API
- 结构化的 `PredictionResult` 推理输出 schema
- 可配置的 ensemble 推理与 `ensemble_forces` 输出
- 面向 adapter 的输入校验、异常处理与配置化开关
- 在线监督相关测试与文档补充

`v0.1` 首次打通的核心能力仍然保留，包括：

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

稳定 Python API：

```python
from gmd_se3gnn.api import train

best_checkpoint = train(
    dataset_path="data/train.extxyz",
    train_config="configs/default.yaml",
    output_dir="outputs/run_api",
)
print(best_checkpoint)
```

### 3. 运行 smoke test

```bash
python scripts/smoke_test.py
```

### 4. 导出 TorchScript 模型

```bash
python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt --output model.pt
```

稳定 Python API：

```python
from gmd_se3gnn.api import export_model

artifact_path = export_model(
    model_path="outputs/run/ckpt_best.pt",
    output_dir="exports",
    export_config={"device": "cpu", "filename": "model.pt"},
)
print(artifact_path)
```

### 5. Python 推理

```python
from gmd_se3gnn.api import OnlinePredictor
import numpy as np

predictor = OnlinePredictor.from_checkpoint(
    "outputs/run/ckpt_best.pt",
    {
        "online_monitoring": {
            "enabled": True,
            "return_energy": True,
            "return_ensemble_forces": False,
            "return_latent_descriptor": False,
            "return_unsafe_probability": False,
            "batch_size": 1,
            "device": "cpu",
            "ensemble": {
                "enabled": False,
                "members": None,
                "checkpoint_paths": [],
            },
        }
    },
)

result = predictor.predict(
    {
        "positions": np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
        "species": np.array([6, 1], dtype=np.int64),
    }
)
print(result.energy)
print(result.forces.shape)
print(result.ensemble_forces)
```

### 6. ASE 接口

```python
from gmd_se3gnn.inference import MLIPCalculator

calc = MLIPCalculator.from_checkpoint("outputs/run/ckpt_best.pt", device="cpu")
atoms.calc = calc.get_ase_calculator()
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## 在线监督接口

推荐外部 adapter 直接调用 `gmd_se3gnn.api`：

```python
from gmd_se3gnn.api import OnlinePredictor, export_model, train
```

关键接口签名：

```python
train(dataset_path, train_config, output_dir, resume_checkpoint=None) -> str
export_model(model_path, output_dir, export_config=None) -> str
predict(checkpoint_path, structure, predict_config=None) -> PredictionResult
OnlinePredictor.from_checkpoint(checkpoint_path, predict_config=None)
OnlinePredictor.predict(structure) -> PredictionResult
OnlinePredictor.predict_batch(structures) -> list[PredictionResult]
```

`PredictionResult` 的稳定输出 schema：

```python
{
  "energy": float | None,
  "forces": np.ndarray,                 # shape (N, 3)
  "ensemble_forces": np.ndarray | None, # shape (M, N, 3)
  "latent_descriptor": np.ndarray | None,
  "unsafe_probability": float | None,
  "metadata": dict[str, Any],
}
```

字段说明：

- `forces` 为必需字段，shape 为 `(N, 3)`
- `energy` 默认返回；关闭后稳定返回 `None`
- `ensemble_forces` 在开启 committee 且请求返回时给出，shape 为 `(M, N, 3)`
- `latent_descriptor` 与 `unsafe_probability` 当前为预留扩展点；若模型尚未提供相应能力，返回 `None`
- `metadata` 包含 checkpoint、device、ensemble 成员数、请求输出项等信息

开启 ensemble 输出示例：

```python
predictor = OnlinePredictor.from_checkpoint(
    "outputs/run_a/ckpt_best.pt",
    {
        "online_monitoring": {
            "enabled": True,
            "return_ensemble_forces": True,
            "ensemble": {
                "enabled": True,
                "members": 2,
                "checkpoint_paths": [
                    "outputs/run_a/ckpt_best.pt",
                    "outputs/run_b/ckpt_best.pt",
                ],
            },
        }
    },
)
```

## 配置建议

- `configs/default.yaml`：通用默认训练配置
- `configs/water.yaml`：水体系示例配置

新增的在线监督配置项：

```yaml
online_monitoring:
  enabled: false
  return_energy: true
  return_ensemble_forces: false
  return_latent_descriptor: false
  return_unsafe_probability: false
  batch_size: 1
  device: cpu
  ensemble:
    enabled: false
    members: null
    checkpoint_paths: []
```

说明：

- 所有在线监督相关能力都可以通过配置启用或关闭
- `ensemble.enabled=true` 时，必须提供至少 2 个 checkpoint
- `batch_size` 用于限制 `predict_batch()` 的最大输入规模
- 当前没有现成 latent / unsafe head 时，字段稳定返回 `None`

如果是第一次跑，建议先从较小模型和较小 batch 开始，先确认数据、训练和导出链路都能跑通。


## License

本项目采用仓库内 `LICENSE` 文件所声明的许可证。
