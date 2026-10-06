# GMD-SGT v0.3

GMD-SGT 是一个面向原子体系建模的 SE(3)/E(3) 等变机器学习势能项目，目标是同时支持训练、推理、TorchScript 导出，以及与外部分子动力学程序的集成。

## Release Note

当前版本：`v0.3`（包版本 `0.3.0`，见 `gmd_sgt.__version__`）

`v0.3` 是一次正确性修复版本，重点包括：

- 修复安装：合法的构建后端，`gmd-train` 入口位于包内（`gmd_sgt.cli`）
- 修复 Stage 1 / Stage 2 训练：力的计算图在训练时保留，能量 + 力联合损失可以正常反向传播
- 等变特征初始化只写入 `0e` 标量通道；球谐 irreps 由 `l_max` 推导并做一致性校验
- 无 e3nn 时：默认（非标量 irreps）模型明确报错；标量 irreps 使用依赖距离的不变消息传递
- 周期性：使用 ASE `primitive_neighbor_list`，按轴 PBC（如 `[True, True, False]`）贯穿数据读取、collate、训练、推理与 ASE
- 截断处平滑：所有消息与 attention 权重都乘以截断包络，能量与力在截断处连续
- TorchScript 导出：`torch.jit.script`（不再回退到 tracing），导出后重新加载并与 eager 的能量与力逐一比对
- 空验证集有明确行为；stress 已实现（`compute_stress=True`，`w_stress > 0`）

兼容性说明见下文“兼容性”一节。

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
- 长程机制可选：支持 `none`、`invariant_attention`、`equivariant_attention`、`electrostatic`（语义见“长程模块”一节）
- 训练与部署衔接清晰：训练完成后可以直接导出为 TorchScript
- 对接外部模拟程序友好：支持外部传入 `edge_index` 和 `edge_shift`

## 目录结构

```text
gmd_sgt/
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
pip install -e .            # 最小依赖：torch, numpy, ase, pyyaml
pip install -e .[e3nn]      # 默认 UnifiedEquivariantMLIP 需要 e3nn
```

如果需要开发与测试依赖：

```bash
pip install -e .[dev]
```

安装后可以直接使用命令行入口：

```bash
gmd-train --config configs/default.yaml
```

依赖说明：

- `UnifiedEquivariantMLIP` 的默认 irreps（`128x0e + 64x1o + 32x2e`）需要 e3nn；未安装时构造模型会直接抛出 `ImportError`。
  未安装 e3nn 时只支持单个纯标量项 irreps（例如 `irreps: "128x0e"`，通道数 `>= scalar_dim`），此时消息传递是依赖距离的 E(3) 不变卷积。
  该路径用纯 PyTorch 精确复现 e3nn 在标量 irreps 下的张量积、Linear 与 BatchNorm（参数名与数值一致），所以标量 irreps 的 checkpoint 在安装与未安装 e3nn 的环境之间可以互相加载。
- `AllegroStyleBackbone` / `GMDSGTModel` 在 `l_max <= 2` 时不依赖 e3nn（闭式球谐与 e3nn 数值一致）；`l_max > 2` 需要 e3nn。

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
from gmd_sgt.api import train

best_checkpoint = train(
    dataset_path="data/train.extxyz",
    train_config="configs/default.yaml",
    output_dir="outputs/run_api",
)
print(best_checkpoint)
```

### 训练细节

- 恢复训练（`--resume` / `resume_checkpoint`）：模型由 checkpoint 中的 `model_config` 与权重重建；配置文件中的模型参数必须与之一致（`atomic_energies` 除外，取自 checkpoint），否则在训练开始前报错。训练参数（输出目录、epoch 数、学习率等）可以修改。
- 数据划分：`val_fraction`、`test_fraction` 必须在 `[0, 1)` 且两者之和 `< 1`。`val_fraction > 0` 时只要数据量允许，至少保留 1 个验证结构。
- 空验证集（例如 `val_fraction: 0`）：跳过验证，`ckpt_best.pt` 与 early stopping 改为依据训练损失，并给出警告；checkpoint 中的 `selection_metric` 记录所用指标（`val_total` 或 `train_total`）。
- stress：`w_stress > 0` 时训练会计算 `stress = (1/V) dE/dε`（ASE 符号约定，eV/Å³），要求每个结构都有 stress 标签和周期性晶胞，否则直接报错。npz 中的 `virial` 会按 `stress = -virial / V` 转换。
- PBC：extxyz / npz 中的按轴周期性（例如表面 `[True, True, False]`）会被保留；只给出单个布尔值时视为三个方向相同。

### 2.1 Stage 1: Allegro-style local backbone

最小 staged residual-learning 改造已先落地 Stage 1。该路径不会替换现有 `UnifiedEquivariantMLIP` 训练入口，而是新增一个独立入口：

```bash
python -m gmd_sgt.training.train_backbone --config configs/stage1_backbone.yaml
```

dry-run 小数据示例：

```bash
python -m gmd_sgt.training.train_backbone \
  --config configs/stage1_backbone.yaml \
  --dry-run
```

当前 Stage 1 特性：

- 轻量 local `Allegro-style` backbone
- Bessel radial basis + 方向基函数编码
- 仅输出总能量 / 原子能
- forces 统一由总能量自动微分得到
- 复用现有 dataset / loss / trainer / checkpoint 机制

后续 Stage 2-5 的 residual / committee / distillation 将继续在该 backbone 之上扩展，而不破坏现有默认训练链路。

Stage 2（在 Stage 1 checkpoint 上训练残差分支，支持 `freeze_backbone` / `semi_freeze_backbone`）：

```bash
python -m gmd_sgt.training.train_residual --config configs/stage2_residual.yaml
```

### 3. 运行 smoke test

```bash
python scripts/smoke_test.py
```

### 4. 导出 TorchScript 模型

```bash
python scripts/export_model.py --checkpoint outputs/run/ckpt_best.pt --output model.pt
```

支持的模型类型：`UnifiedEquivariantMLIP`（全部 `long_range_type`）、`AllegroStyleBackbone`、`GMDSGTModel`。
导出使用 `torch.jit.script`，保存后会重新加载，并在多种原子数 / 边数（包括空边列表和周期性平移）的结构上与 eager 模型比较能量和力；任何不一致都会抛出 `ExportValidationError`。

导出模型的接口保持不变：

```text
forward(species[N] int64, positions[N,3] float32,
        edge_index[2,E] int64, edge_shift[E,3] float32)
  -> {"energy": [1] float64, "forces": [N,3] float32}
```

约定 `r_ij = pos[edge_index[1]] - pos[edge_index[0]] + edge_shift`，需要提供 `local_cutoff` 内每对原子的两个方向（超出截断的边权重为零，等价于不存在）。

`electrostatic` 模型只适用于非周期体系：导出模型在任何 `edge_shift` 非零时报错。但四张量接口无法表达所有周期性情况——例如晶胞大于两倍截断、邻居表中恰好没有镜像边，或以零平移传入未折叠的坐标——这些输入无法被检测，会被当作孤立团簇计算，需要调用方保证不传入周期体系。
力由 autograd 计算，调用时不能处于 `torch.no_grad()` / `NoGradGuard` / inference mode。导出接口不包含 stress。

稳定 Python API：

```python
from gmd_sgt.api import export_model

artifact_path = export_model(
    model_path="outputs/run/ckpt_best.pt",
    output_dir="exports",
    export_config={"device": "cpu", "filename": "model.pt"},
)
print(artifact_path)
```

### 5. Python 推理

```python
from gmd_sgt.api import OnlinePredictor
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
from gmd_sgt.inference import MLIPCalculator

calc = MLIPCalculator.from_checkpoint("outputs/run/ckpt_best.pt", device="cpu")
atoms.calc = calc.get_ase_calculator()
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## 在线监督接口

推荐外部 adapter 直接调用 `gmd_sgt.api`：

```python
from gmd_sgt.api import OnlinePredictor, export_model, train
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

## 长程模块

- `invariant_attention` / `equivariant_attention` 是**结构内全局** attention：同一结构内所有原子互相 attend，与距离无关，也不包含周期性镜像。它们作用在已经由局域（带截断、支持 PBC）消息传递得到的特征上，因此能量平滑，且对原子在晶胞内的平移折叠不变。
- 没有任何长程模块使用 `lr_cutoff`；该参数仅为配置与 checkpoint 兼容而保留（`MLIPCalculator.lr_cutoff` 仅作信息用途）。
- `electrostatic` 对同一结构内所有原子对求屏蔽库仑和，不做 Ewald 求和，只适用于非周期体系；输入周期性结构时会报错。

## 兼容性

- 公共 API 与导出签名保持不变；模型 `forward` 新增可选的 `pbc` 关键字参数（放在参数列表末尾）。
- 旧 checkpoint 的 `state_dict` 仍可加载（未使用的 `radial_basis_lr.freq` 会被忽略），但截断平滑、attention 归一化、特征初始化与 l=2 基函数归一化都已修正，因此预测结果与旧版本不同，加载时会给出警告；生产使用前请重新训练或微调。
- 未安装 e3nn 时训练得到的 `UnifiedEquivariantMLIP` checkpoint（旧的占位实现，能量与几何无关）不能再加载。

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
