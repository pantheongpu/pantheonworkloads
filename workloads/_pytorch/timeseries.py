"""Shared body of the catalog's time-series forecasting workloads (Chronos family), imported by their main.py.

The input is a synthetic series made here with numpy (a trend, a 24-step season and a fixed-seed noise term; 512
points, nothing downloaded). The 24-step median forecast is compared with a tolerance and must follow the season
(correlation with the true continuation above 0.8). Uses the `chronos-forecasting` package
(`BaseChronosPipeline.from_pretrained` on the pinned snapshot, `predict_quantiles`). Bench mode (PW_TS_MODE=bench):
forecasts per second at batch 1. Run on an A10G (stage 2a); the package's calls for Chronos-2 (several variates, group
attention) are the least certain part.
"""
import os
import sys
import time

import numpy as np
import torch

import catalog_common as cc
import common

PRED = 24


def series(n=512 + PRED):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    return 10.0 + 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 24) + 0.2 * rng.standard_normal(n)


def run(model_id, revision, pipeline, label=None, dtype="bfloat16"):
    mode = os.environ.get("PW_TS_MODE", "functional")
    if common.DEVICE == "cpu":
        common.skip("this time-series model is run on a GPU only")
    common.setup()
    path = cc.fetch_snapshot(model_id, revision)
    try:
        from chronos import BaseChronosPipeline
    except ImportError as e:
        common.skip(f"chronos-forecasting is not installed ({e}); pip install -r workloads/_pytorch/requirements-chronos.txt")
    t0 = time.perf_counter()
    pipe = BaseChronosPipeline.from_pretrained(path, device_map=common.DEVICE, torch_dtype=getattr(torch, dtype))
    load_s = time.perf_counter() - t0
    full = series()
    ctx, truth = full[:-PRED], full[-PRED:]

    def go():
        context = torch.tensor(ctx, dtype=torch.float32)
        if pipeline == "chronos2":
            context = context.view(1, 1, -1)
        quantiles, mean = pipe.predict_quantiles(context, prediction_length=PRED, quantile_levels=[0.1, 0.5, 0.9])
        q = quantiles[0] if isinstance(quantiles, (list, tuple)) else quantiles   # one series in: its (variates x) PRED x 3 quantiles
        q = torch.as_tensor(q).float().reshape(-1, PRED, 3)[0]
        return q[:, 1].cpu().numpy()

    median = go()
    corr = float(np.corrcoef(median, truth)[0, 1])
    if not np.isfinite(median).all() or corr < 0.8:
        sys.exit(f"the forecast does not follow the season: correlation {corr:.3f}, values {median[:6]}")
    out = {"median_forecast": [round(float(x), 2) for x in median]}
    metrics = {}
    if mode == "bench":
        for _ in range(3):
            go()
        common.sync()
        t = time.perf_counter()
        n = 20
        for _ in range(n):
            go()
        common.sync()
        metrics = {"forecasts_per_s_b1": round(n / (time.perf_counter() - t), 2)}
        out = f"{label or model_id} bench ok"
    common.finish(out, f"{common.device_name()}, torch {torch.__version__}, {pipeline}, load {load_s:.0f} s, correlation with the true continuation {corr:.3f}", metrics)
