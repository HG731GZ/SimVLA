# SimVLA 复现版本差异报告

## 1. 总体结论

源码层面共有 14 个已跟踪文件被修改，统计为约 964 行新增、292 行删除。主要改动方向是：

- 增强 LIBERO 数据集 metadata / norm 统计生成脚本，加入完整性校验、坏文件检查和 `libero_100/libero_10`、`libero_100/libero_90` 分组目录兼容。
- 将 norm state 方向从原先容易混淆的 Euler 统计改为与训练 handler 一致的 axis-angle 统计。
- 训练脚本增加梯度累积、可配置 SmolVLM dtype、可配置序列长度、局部预训练模型路径和更灵活的多卡参数。
- 评估端增加 task 子集、max steps 覆盖、动作裁剪、LIBERO 路径配置和本地 checkpoint 加载兼容。
- 当前目录中还包含大量运行产物：LIBERO HDF5 数据、预训练权重、checkpoint、评估视频、IDE 配置、Python 缓存等。这些不建议作为源码改动提交。

总体判断：核心代码改动大多有实际复现价值，建议保留并整理；数据、权重、日志、视频、缓存类文件建议删除或加入 `.gitignore`，不要进入代码版本管理。

## 2. 差异概览

已跟踪源码修改：

| 类型 | 文件 | 主要变化 | 建议 |
|---|---|---|---|
| 数据校验 | `create_libero_meta.py` | 支持 grouped LIBERO-100 目录、检查 HDF5 必需字段、增加 `--validate_only` / `--skip_bad_files` / `--allow_incomplete` | 保留，但与 norm 脚本抽公共工具 |
| norm 统计 | `compute_libero_norm_stats.py` | 增加数据 preflight、官方数量校验、Euler 到 axis-angle 转换、metadata 记录 | 保留，axis-angle 与训练 handler 一致 |
| 数据读取 | `datasets/dataset_smolvlm.py` | 遇到轨迹读取错误时 fail-fast，报告具体路径 | 保留，便于定位坏文件 |
| 数据路径 | `datasets/domain_handler/libero_hdf5.py` | 自动解析 stale flat `libero_10` / `libero_90` 路径到 `libero_100/...`，并检查缺失文件 | 保留，但最好通过重新生成 metadata 根治 |
| 模型配置 | `models/configuration_smolvlm_vla.py` | 新增 `vlm_torch_dtype` | 保留 |
| 模型加载 | `models/modeling_smolvlm_vla.py` | 支持 `float32` / `float16` / `bfloat16` / `auto` 加载 VLM backbone | 保留 |
| 训练主程序 | `train_smolvlm.py` | 加入梯度累积、effective batch 记录、dtype / max_len_seq 参数、freeze 阶段动态关梯度 | 保留，但有细节可优化 |
| 训练脚本 | `train_smolvlm_large.sh`, `train_smolvlm_small.sh` | 改为本地 SmolVLM、刷新 metadata、校验 norm step、环境变量控制 GPU/进程/端口 | 保留，但需要整理 env override |
| 评估 client | `evaluation/libero/libero_client.py` | 加入 `--task_ids`、`--max_steps`、默认 action clip | 保留 |
| 评估 server | `evaluation/libero/serve_smolvlm_libero.py` | 手动从 config + safetensors/bin 加载 checkpoint，可覆盖 SmolVLM backbone 路径 | 保留，但建议更严格地处理 missing/unexpected keys |
| 评估脚本 | `evaluation/libero/run_eval_all.sh` | 支持外部 `LIBERO_ROOT` 和 `LIBERO_CONFIG_PATH` | 保留 |
| 生成文件 | `datasets/metas/libero_train.json`, `norm_stats/libero_norm.json` | 按 grouped LIBERO-100 目录和 axis-angle norm 重新生成 | 不建议长期手写维护，建议由脚本生成 |

## 3. 关键功能差异

### 3.1 LIBERO 数据集完整性校验

`create_libero_meta.py` 和 `compute_libero_norm_stats.py` 现在都会做 HDF5 preflight：

- 检查 `data/demo_*` 是否存在。
- 检查 `actions`、`obs/agentview_rgb`、`obs/eye_in_hand_rgb`、`obs/ee_pos`、`obs/ee_ori`、`obs/gripper_states` 等关键字段。
- 默认要求官方 full LIBERO split 数量匹配：
  - `libero_10`: 10 files / 500 demos
  - `libero_goal`: 10 files / 500 demos
  - `libero_object`: 10 files / 500 demos
  - `libero_spatial`: 10 files / 500 demos
  - `libero_90`: 90 files / 4500 demos
  - full steps: 1007618

这是有价值的改动。官方代码原先对坏文件和缺失目录更宽松，容易在复现时悄悄跳过数据，最后得到看似能跑但统计不一致的结果。

可优化点：

- `resolve_subset_dirs`、expected counts、HDF5 key 校验逻辑在两个脚本里重复，建议抽到 `datasets/libero_validation.py` 或 `datasets/domain_handler/libero_utils.py`。
- `EXPECTED_FULL_LIBERO_STEPS=1007618` 是强约束，适合 full split；如果经常做 subset 实验，可以把它改成只在请求完整五个子集时启用，或提供更清晰的 `--strict_full_split`。
- Euler 到 axis-angle 目前逐条循环转换，可以用 `Rotation.from_euler(...).as_rotvec()` 简化和加速。

### 3.2 metadata 路径布局变更

官方 metadata 使用 flat 路径，例如：

```text
./datasets/metas/libero_10/...
./datasets/metas/libero_90/...
```

当前版本改成支持 grouped 布局：

```text
./datasets/metas/libero_100/libero_10/...
./datasets/metas/libero_100/libero_90/...
```

这个改动合理，因为很多 LIBERO 下载包会把 `libero_10` 和 `libero_90` 放在 `libero_100` 下。`datasets/domain_handler/libero_hdf5.py` 还加入了 fallback resolver，可以把旧 flat metadata 自动映射到 grouped 目录。

当前工作区风险：

- `datasets/metas/libero_train.json` 声称有 130 个 HDF5 / 6500 demos。
- 实际在当前工作区能找到的只有 10 个路径，缺失 120 个路径。
- `datasets/metas/libero_spatial_train.json` 的 10 个路径都存在。

因此，如果现在直接用 `datasets/metas/libero_train.json` 训练，修改后的 handler 会主动报错，这是正确行为。要训练 full split，需要补齐 `libero_goal`、`libero_object`、`libero_100/libero_10`、`libero_100/libero_90` 数据；要只训练 spatial，则应显式使用 `datasets/metas/libero_spatial_train.json` 和对应 norm。

### 3.3 norm stats 改动

`norm_stats/libero_norm.json` 当前 metadata 显示：

- `num_demos`: 6500
- `num_steps`: 1007618
- `state_orientation_format`: `axis_angle`
- `resolved_subsets` 包含 `libero_goal`、`libero_object`、`libero_spatial`、`libero_10`、`libero_90`

这个方向与 `LiberoHDF5Handler` 的训练输入一致：handler 会把 HDF5 中的 Euler `obs/ee_ori` 转为 axis-angle 后拼入 proprio。因此，axis-angle norm 是更一致的选择。

需要注意：

- `norm_stats/libero_spatial_norm.json` 仍然是旧格式，metadata 没有 `state_orientation_format`，state labels 还是 `ori_r/ori_p/ori_y`。如果它是旧版脚本生成的，建议用当前脚本重新生成，或者删除以避免误用。
- `norm_stats/official/` 里有两份小型 official norm 备份。如果需要对照，可以保留；如果只是临时备份，建议移到外部实验记录目录。

### 3.4 训练流程改动

`train_smolvlm.py` 新增：

- `--gradient_accumulation_steps`
- `--max_len_seq`
- `--vlm_torch_dtype`
- Accelerator 中启用 gradient accumulation。
- 日志记录 effective global batch size。
- freeze 阶段动态设置 VLM 和 transformer core 的 `requires_grad`。
- Tensor 移动改为 `.to(accelerator.device)`，比硬编码 `.cuda()` 更通用。

这些都是有价值的复现改动，尤其适合单卡或显存受限环境。

可优化点：

- `train_smolvlm.py` 存在两处尾随空格，`git diff --check` 报在第 432 和 443 行。
- shell 脚本里 `NUM_WORKERS=4`、`LOG_INTERVAL=20` 仍是硬编码；但 `训练指令` 文件里导出了 `NUM_WORKERS=8`、`LOG_INTERVAL=100`，实际不会生效。建议改为 `NUM_WORKERS=${NUM_WORKERS:-4}`、`LOG_INTERVAL=${LOG_INTERVAL:-20}`。
- shell 脚本调用 `python`，当前裸环境没有 `python` 命令，只有 `python3`。conda 环境中可能正常，但可移植写法可以改成 `PYTHON=${PYTHON:-python}`，或在文档中明确必须在 conda 环境执行。
- `create_libero_meta.py` 现在每次训练都会刷新 metadata，这是安全的；如果数据集很大且频繁启动，可以加 `--validate_only` 或 checksum 缓存来减少启动成本。

### 3.5 模型加载和 dtype

`models/configuration_smolvlm_vla.py` 和 `models/modeling_smolvlm_vla.py` 支持通过 config/CLI 控制 VLM backbone dtype。默认仍是 `float32`，与官方训练稳定性取向一致；需要省显存时可以尝试 `bf16` 或 `fp16`。

评估 server 不再直接 `SmolVLMVLA.from_pretrained(checkpoint_path)`，而是：

1. 从 checkpoint 读取 `SmolVLMVLAConfig`。
2. 可覆盖 `config.smolvlm_model_path` 指向本地 `pretrained/SmolVLM-500M-Instruct`。
3. 手动加载 `model.safetensors` 或 `pytorch_model.bin`。
4. `strict=False` 加载权重。

这个改动有助于本地离线评估 official checkpoint。建议保留。

可优化点：

- 当前只打印 missing/unexpected key 的数量，不打印具体 key。建议至少打印前 20 个，或者在数量非零时提供 `--allow_partial_load`，默认严格失败。
- 如果 checkpoint config 的 dtype 与 CLI/环境想要的 dtype 不一致，可以给 server 也加 `--vlm_torch_dtype`。

### 3.6 LIBERO 评估改动

`evaluation/libero/libero_client.py` 新增：

- `--task_ids`：可以只跑某些 task，适合 smoke test。
- `--max_steps`：覆盖 rollout 最大步数。
- 默认对 action 做 `np.clip(action, -1, 1)`，并提供 `--no_clip_actions` 关闭。

这些都适合复现调试，建议保留。

可优化点：

- `WebSocketClient` 内部默认 `resize_size=224`，CLI 没有暴露。训练和 server 默认 image size 是 384，虽然 server 会再次 resize，但建议加 `--resize_size`，避免无意先降采样再升采样。
- action clipping 默认开启能防止环境崩，但也可能掩盖 norm/action scale 问题。正式报告指标时建议记录是否启用了 clipping。

## 4. 未跟踪文件与目录

当前版本有大量未跟踪文件。按用途分组如下：

| 路径 | 大小/数量 | 判断 | 建议 |
|---|---:|---|---|
| `datasets/metas/libero_spatial/` | 约 5.9G，10 个 HDF5 + `.DS_Store` | 数据集内容 | 不进 Git；移到数据盘或用 DVC/Git LFS 管理 |
| `datasets/metas/libero_spatial.zip` | 约 2.7G | 数据压缩包 | 不进 Git；如果已解压可删除或外部保存 |
| `pretrained/SmolVLM-500M-Instruct/` | 约 6.5G | 预训练模型 | 不进 Git；保留本地路径，加入 ignore |
| `runs/` | 约 9.7G | 训练 checkpoint、official checkpoint、日志 | 不进 Git；只保留外部实验记录或上传模型仓库 |
| `evaluation/libero/eval_results/` | 388 个 mp4，约 91M | 评估视频 | 不进 Git；需要时只保留成功率 summary |
| `__pycache__/`, `datasets/**/__pycache__/`, `models/__pycache__/`, `evaluation/**/__pycache__/` | 多个 pyc | Python 缓存 | 可删除并 ignore |
| `.idea/` | 约 11K | IDE 配置 | 可删除或 ignore |
| `=4.57.0` | 约 12K | 看起来是误把 `pip install transformers>=4.57.0` 的输出重定向到了文件 | 可删除 |
| `loss_visualizer.html` | 约 44K | 可视化工具/报告 | 如果有用，建议移到 `tools/` 或 `reports/` 并说明用途；否则删除 |
| `训练指令`, `评估指令-official`, `评估指令（复件）` | 小文本 | 本地操作笔记 | 建议改名为 `.md`，放到 `docs/`；若仅个人临时笔记则不提交 |
| `evaluation/libero/.libero_config/config.yaml` | 很小 | 本地 LIBERO 配置 | 通常不提交，避免绑定个人路径 |
| `norm_stats/libero_spatial_norm.json` | 很小 | spatial-only norm | 如果继续做 spatial 子集实验可保留，但建议重新生成新版 metadata；否则删除 |
| `norm_stats/official/` | 很小 | official norm 备份 | 可保留为对照，但建议写清来源；重复副本可删 |

建议新增 `.gitignore`，至少包含：

```gitignore
__pycache__/
*.py[cod]
.idea/
.DS_Store
=*

datasets/metas/**/*.hdf5
datasets/metas/**/*.zip
pretrained/
runs/
evaluation/libero/eval_results/
evaluation/libero/.libero_config/
loss_visualizer.html
```

## 5. 可删除 / 可优化清单

优先可删除或移出仓库：

- `=4.57.0`
- `.idea/`
- 所有 `__pycache__/` 和 `*.pyc`
- `datasets/metas/libero_spatial/.DS_Store`
- `datasets/metas/libero_spatial.zip`，如果已解压且外部有备份
- `evaluation/libero/eval_results/` 下的视频，若不再需要逐帧复查
- `runs/` 和 `pretrained/`，如果目标是保持源码仓库干净

建议保留但优化：

- `create_libero_meta.py` / `compute_libero_norm_stats.py` 的校验逻辑：保留，但抽公共模块。
- `datasets/domain_handler/libero_hdf5.py` 的路径 fallback：保留，但长期应依赖正确生成的 metadata。
- `train_smolvlm.py` 的梯度累积和 dtype：保留，清理尾随空格。
- `train_smolvlm_*.sh`：保留，但让 `NUM_WORKERS`、`LOG_INTERVAL` 等参数支持环境变量覆盖。
- `evaluation/libero/*`：保留 smoke-test 功能，但补充 `--resize_size` 和更严格 checkpoint key 检查。

建议谨慎处理：

- `datasets/metas/libero_train.json`：当前 workspace 缺失 120/130 个 HDF5 路径。full split 数据补齐前不应作为可运行配置使用。
- `norm_stats/libero_norm.json`：它记录 full split 的 1007618 steps，但当前 workspace 没有对应 full dataset。若只是从另一台机器生成的统计，可以保留作实验输入；否则建议在当前数据集上重新生成。
- `norm_stats/libero_spatial_norm.json`：旧 metadata 风格，方向字段不明确，建议重生成。

## 6. 验证结果

已执行的轻量检查：

```text
python3 -m py_compile compute_libero_norm_stats.py create_libero_meta.py train_smolvlm.py \
  datasets/dataset_smolvlm.py datasets/domain_handler/libero_hdf5.py \
  models/configuration_smolvlm_vla.py models/modeling_smolvlm_vla.py \
  evaluation/libero/libero_client.py evaluation/libero/serve_smolvlm_libero.py
```

结果：通过。

```text
git diff --check
```

结果：失败，原因是 `train_smolvlm.py:432` 和 `train_smolvlm.py:443` 有尾随空格。

metadata 路径检查：

```text
datasets/metas/libero_train.json: num_files=130, existing_paths=10, missing_paths=120
datasets/metas/libero_spatial_train.json: num_files=10, existing_paths=10, missing_paths=0
```

这说明当前目录更像是 spatial 子集 + full split 配置/统计/模型产物的混合状态。为了复现稳定，建议先明确当前实验目标：

- 只复现 `libero_spatial`：使用 `libero_spatial_train.json` 和重新生成的 spatial norm。
- 复现 full LIBERO：补齐所有五个子集的数据，再使用 `libero_train.json` 和 `libero_norm.json`。

