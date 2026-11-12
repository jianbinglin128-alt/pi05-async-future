# 当前观测与单帧固定 RoPE 修复

日期：2026-11-08。仓库：`/data1/lin/Apaper2/openpi-copy`。

## 代码行为

- `sample_actions_streaming` 每次请求都编码本次图像，等待视觉 KV 发布后再建立去噪快照。
  `concurrent_vision=True` 使用视觉 worker，`False` 使用调用线程，两者均不再读取上一轮请求的视觉。
- `StreamingPrefixCache(max_frames=1)` 在每次覆写前，将视觉起始位置恢复到各 batch 样本的
  有效语言 token 数。新帧及动作 suffix 使用这次 mask 对应的固定坐标，不再随请求次数漂移。
- 单帧 ring、decode snapshot 以及 CUDA Graph 的输入存储仍可复用。多帧窗口原位置策略保留。
- broker 仍每次执行 5 步、提前 2 步提交推理，观测序列为 O3、O8、O13……。
  下一 chunk 开始时最新图像落后 2 个控制步；20 Hz 下对应 100 ms 的控制时间。
  推理来不及完成时会在边界等待。
- 保留分区注意力和固定语言 KV。本次只修复用户指定的视觉滞后与单帧位置漂移，未恢复原模型的
  图像语言联合前缀，未加入新的训练或状态条件。

## 验证

20 项回归测试通过，包括 CPU/CUDA cache、CUDA Graph、当前图像 worker 对照、broker、
WebSocket reset 和 LIBERO 状态预测。新增回归覆盖不同 batch mask、图像变化、存储地址复用、
跨请求 graph 复用，以及 O3/O8/O13 提交与执行边界。修改文件的 Ruff 检查通过。

真实权重配对评测共 60 个 episode 已正常结束，0 异常；进程已退出，GPU 6 已释放。

复现命令（从仓库根目录运行；换一个新的 output 目录，避免追加旧 JSONL）：

```bash
CUDA_VISIBLE_DEVICES=6 JAX_PLATFORMS=cpu \
OPENPI_DATA_HOME=/data1/lin/openpi_cache \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
PYTHONPATH=.:/data1/lin/Apaper2/openpi/third_party/libero \
.venv/bin/python scripts/diagnose_reflex.py \
  --checkpoint /data1/lin/openpi_cache/openpi-assets/checkpoints/pi05_libero_pytorch \
  --output data/libero/reflex_fixed_single_frame_20261108 \
  --tasks 0 1 2 3 4 5 6 7 8 9 --trials 3 \
  --modes standard partition_fixed1
```

两组均使用同一个 checkpoint、CUDA Graph、每组执行 5 步与提前 2 步的异步 broker，
各任务使用官方初始状态 0、1、2，环境 seed 7，噪声为 `42 + inference_index`，每 episode 重置。
初始图像与状态的 SHA256 用于核对成对输入。`partition_fixed1` 使用单帧窗口与视觉 worker。
forensics 会启用零输出状态分支，两组保持相同配置。

本轮是本地 policy 与 broker 的闭环仿真，环境渲染和模型均在 GPU 6，无 WebSocket 网络开销，
仿真未按 20 Hz 实时节拍限速。推理耗时不能直接与此前分 GPU 的服务端评测作性能对比。

## 闭环成功率

| 配置 | 成功 / episode | 成功率 |
| --- | ---: | ---: |
| 标准视觉语言联合前缀 + 异步 broker | 27 / 30 | 90.0% |
| 修复：本次图像 + 单帧固定 RoPE + 异步 broker | 23 / 30 | 76.7% |

之前旧视觉 + 两帧历史的评测为 1/30（3.3%）。该旧评测没有固定 flow noise，窗口也不同，
因此只能作为历史参照，不能把 73.4 个百分点的差值全部归因于本轮两处修复。
本轮标准组与修复组之间为严格配对：30 对初始图像/状态哈希全部一致。
21 对两组均成功，6 对仅标准成功，2 对仅修复成功，1 对均失败。
修复组比标准组少成功 4 次（13.3 个百分点）；30 次/组不足以代表完整 benchmark。

| 任务 ID | 标准成功 / 3 | 修复成功 / 3 |
| --- | ---: | ---: |
| 0 | 3 | 3 |
| 1 | 3 | 2 |
| 2 | 2 | 3 |
| 3 | 3 | 3 |
| 4 | 3 | 2 |
| 5 | 2 | 3 |
| 6 | 3 | 3 |
| 7 | 3 | 1 |
| 8 | 3 | 3 |
| 9 | 2 | 0 |

任务 7、9 是本轮差异最大的任务；此结果尚不能确定剩余失败的单一原因。
固定语言 KV 与分区注意力仍改变了原 checkpoint 的计算语义，需独立消融。

## 数值与耗时

相同图像、提示词和噪声重复推理 45 次，所记录的第 2、5、10、20、30、45 次
动作最大差与 RMSE 均为 0，位置保持在 532。
任务 0、6、8 的第一个初始状态，生产修复路径与上一轮手动单帧 rebase 诊断的完整动作序列
最大差均为 0，分别在 93、126、110 步成功。
修复组所有记录的 `vision_snapshot_lag` 均为 0。

以下为去掉每个 episode 首次推理后的单次耗时中位数，单位 ms：

| 阶段 | 标准 | 修复 |
| --- | ---: | ---: |
| 模型推理（含输出 CPU materialization） | 124.85 | 120.35 |
| 视觉与前缀 | 97.57 | 90.54 |
| 动作专家（全部 10 步去噪） | 26.28 | 27.88 |

模型推理均值分别为 126.08 / 121.01 ms，P95 分别为 144.89 / 131.99 ms。
修复组在本轮条件下略快于标准组，但约 120 ms 的中位数超过 20 Hz 下两步的 100 ms 预算。
broker 保留边界等待保证正确性；这组同 GPU、未限速仿真不能证明真实 20 Hz 运行可以完全隐藏推理。

## 证据文件

- `data/libero/reflex_fixed_single_frame_20261108/summary.json`：成功率、逐任务、配对和耗时汇总。
- `data/libero/reflex_fixed_single_frame_20261108/episodes.csv`：60 个 episode 简表。
- `data/libero/reflex_fixed_single_frame_20261108/ablations.jsonl`：原始记录与每次推理耗时。
- `data/libero/reflex_fixed_single_frame_20261108/prefix_forensics.json`：固定输入检查。
- `data/libero/reflex_fixed_single_frame_20261108/actions_task*_trial*_*.npy`：完整动作序列。
- `data/libero/reflex_fixed_single_frame_20261108/run.json`：运行配置与源码哈希。
- `data/libero/reflex_fixed_single_frame_20261108/eval.log`：完整日志。
