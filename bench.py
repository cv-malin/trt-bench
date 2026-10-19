#!/usr/bin/env python3
"""ONNX Runtime 推理基准：延迟分位数与吞吐。

    python bench.py --model model.onnx --provider cuda --runs 200

provider: cpu / cuda / tensorrt
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("需要 onnxruntime：pip install onnxruntime（GPU 用 onnxruntime-gpu）")

PROVIDERS = {
    "cpu": [("CPUExecutionProvider", {})],
    "cuda": [("CUDAExecutionProvider", {}), ("CPUExecutionProvider", {})],
    "tensorrt": [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./.trt_cache",
        }),
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ],
}


def build_session(model: str, provider: str, fp16: bool) -> ort.InferenceSession:
    if provider not in PROVIDERS:
        sys.exit(f"不支持的 provider: {provider}（可选 {', '.join(PROVIDERS)}）")

    options = []
    for name, opts in PROVIDERS[provider]:
        opts = dict(opts)
        if name == "TensorrtExecutionProvider":
            opts["trt_fp16_enable"] = bool(fp16)
        options.append((name, opts))
    return ort.InferenceSession(model, providers=options)


def resolve_shape(sess: ort.InferenceSession, override: str | None) -> list[int]:
    if override:
        return [int(x) for x in override.split(",")]

    spec = sess.get_inputs()[0]
    shape = []
    for d in spec.shape:
        shape.append(1 if isinstance(d, str) else int(d))
    return shape


def measure(sess: ort.InferenceSession, shape: list[int],
            warmup: int, runs: int) -> list[float]:
    name = sess.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main() -> None:
    p = argparse.ArgumentParser(description="ONNX Runtime 推理基准")
    p.add_argument("--model", required=True)
    p.add_argument("--provider", default="cpu", choices=list(PROVIDERS))
    p.add_argument("--shape", default=None, help="如 1,3,640,640；默认按模型静态形状")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--fp16", action="store_true", help="TensorRT EP 启用 FP16")
    args = p.parse_args()

    sess = build_session(args.model, args.provider, args.fp16)
    shape = resolve_shape(sess, args.shape)

    times = np.asarray(measure(sess, shape, args.warmup, args.runs))
    if times.size == 0:
        sys.exit("没有有效测量结果")

    print(json.dumps({
        "provider": args.provider,
        "precision": "fp16" if args.fp16 else "fp32",
        "input_shape": shape,
        "warmup": args.warmup,
        "runs": args.runs,
        "mean_ms": round(float(times.mean()), 3),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "p99_ms": round(float(np.percentile(times, 99)), 3),
        "fps": round(float(1000.0 / times.mean()), 2),
    }, indent=2))


if __name__ == "__main__":
    main()
