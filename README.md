# trt-bench

视觉模型推理基准：同一模型、同一输入，测出**可复现**的延迟分布与吞吐。

大多数"XX ms/帧"的结论都缺少三样东西：硬件型号、预热次数、分位数。这个仓库就是补齐它们的最小工具。

## 用法

```bash
pip install -r requirements.txt

# CPU
python bench.py --model model.onnx --runs 200 --warmup 20

# CUDA
python bench.py --model model.onnx --provider cuda

# TensorRT（需 onnxruntime-gpu + 本机 TensorRT）
python bench.py --model model.onnx --provider tensorrt --fp16
```

输出 JSON：

```json
{
  "provider": "cuda",
  "input_shape": [1, 3, 640, 640],
  "mean_ms": 8.42,
  "p50_ms": 8.21,
  "p95_ms": 9.05,
  "p99_ms": 9.63,
  "fps": 118.7,
  "runs": 200
}
```

## 方法学

测量误差通常比优化收益还大，所以先把测量做对：

1. **预热**：GPU 首次运行包含上下文初始化、kernel 加载，前 10~50 次不计入统计。
2. **分位数**：只报平均值会掩盖长尾。产线关心的是 P99 而不是平均。
3. **同步**：CUDA 是异步的，不同步就 `perf_counter()` 测到的是入队时间。ORT 的 `run()` 内部已同步，自写 TRT 推理时务必 `cudaStreamSynchronize`。
4. **固定输入**：随机数也固定 seed，避免动态 shape 触发重新规划。
5. **隔离变量**：一次只改一个（精度、batch、分辨率、provider）。
6. **记录版本矩阵**：GPU 型号 / 驱动 / CUDA / cuDNN / TensorRT / ORT / 模型哈希。

## 读数建议

| 指标 | 用途 |
| :--- | :--- |
| P50 | 典型帧耗时 |
| P95 / P99 | 卡顿与超时预算 |
| FPS | 单流吞吐上限；多流要乘并发数 |
| 首次延迟 | 冷启动（服务化场景常被忽略） |

## 常见误读

- **"平均 8 ms"但 P99 是 40 ms**：长尾来自调度、显存分配或温度降频。
- **batch=1 和 batch=8 的 FPS 不能直接比**：要看单帧成本是否随 batch 摊薄。
- **不同精度下比较 FPS 却用不同输入分辨率**：结论无效。
- **拿桌面显卡数据推工控机性能**：功耗墙和散热差异巨大。

## TensorRT 引擎基准

`bench.py` 走的是 ORT 的 TensorRT EP。若要测原生 engine（含 INT8 校准），用官方 `trtexec`：

```bash
trtexec --loadEngine=model.engine --shapes=images:1x3x640x640 \
        --warmUp=500 --duration=10 --avgRuns=100
```

`trtexec` 会输出 GPU Compute Time 的 min/mean/median/max 与 90%/95%/99% 分位。

## 已知限制

- 只测推理，不含解码、NMS、预处理；端到端延迟请用 `cv-inspect-kit` 的 `infer --bench`。
- TensorRT EP 首次运行会构建引擎，耗时不计入统计（已单独预热）。
- 未实现多流并发测试，多路视频场景需要扩展。

## 许可

MIT
