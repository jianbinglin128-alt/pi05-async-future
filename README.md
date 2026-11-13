# openpi-future

从 `/data1/lin/Apaper2/openpi-copy` 复制的 future 专用运行副本。

GitHub 仓库：[jianbinglin128-alt/pi05-async-future](https://github.com/jianbinglin128-alt/pi05-async-future)。
从其他机器安装和下载权重，见 [安装与运行说明](docs/future_runtime.md)。
上游模型和通用工具说明见 [README.upstream.md](README.upstream.md)。

固定运行配置：

- 权重：`/data1/lin/Apaper2/model/pi05_libero_async_future`
- 配置名：`pi05_libero_async_future`
- JAX / Orbax，原始整体 JIT 采样，10 次去噪；不启用分阶段 profiler。
- 每次返回形状 `[10, 7]` 的动作 chunk，每轮只执行其前 5 步。
- 执行完 3 步后，使用当前真实图像、真实状态和剩余 2 步排队动作发起异步推理。
- 当前 chunk 的最后 2 步与推理重叠；执行完后等待结果（如尚未完成）。
- 新输出已经对齐到这 2 步之后，直接取新 chunk 的前 5 步，不跳过新输出前 2 步。
- 不预测未来机器人状态，不使用 streaming/reflex 推理缓存。

future 模型本身仍需要 `action_context` 字段表示排队动作。这不是旧 context 权重入口。已移除旧 context 模型配置，future 服务入口校验权重 manifest、SHA256 并严格检查参数树，防止误加载其他权重或忽略额外参数。

## 启动

下面以 GPU 0、端口 18900 为例，先启动服务器，再在另一个终端启动评测。GPU 由调用者指定。

```bash
cd /data1/lin/Apaper2/openpi-future
bash scripts/run_future.sh server 0 18900
```

```bash
cd /data1/lin/Apaper2/openpi-future
bash scripts/run_future.sh eval 0 18900 --suite libero_spatial --trials 3 \
  --output data/libero/future_spatial_3trials
```

四个套件、每任务 50 次：

```bash
bash scripts/run_future.sh eval 0 18900 --suite all --trials 50 \
  --output data/libero/future_four_suites_50trials
```

输出目录必须未使用。当前请求仅复制、配置和验证代码，没有自动启动上述 GPU 评测。

## 计时与验证

客户端端到端推理直接计时 `client.infer`；动作直接计时 `env.step`；任务时间从 reset 和 10 步稳定动作之后，到最后一次 env.step 返回。排除 3 次预热、文件写入和结束后未完成的预取排空。终止时尚未返回的预取记录标为 `terminal_prefetch`。不写视频，不人为限速至 20 Hz。

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m unittest discover -s tests -p 'test_future_schedule.py' -v
JAX_PLATFORMS=cpu .venv/bin/python scripts/check_future_setup.py
```

服务器本地复制来源和验证记录见 `FUTURE_SETUP_AUDIT.json`；这些机器审计文件不上传 Git。
可公开的配置及验证摘要见 [ASYNC_FUTURE_MANIFEST.json](ASYNC_FUTURE_MANIFEST.json)。
原 `openpi-copy` 保留。
