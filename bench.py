"""
Замер скорости инференса на CPU (onnxruntime), как у организатора:
  * время формирования признака одного ТС при батче 1 (кроп -> reid.onnx x2 с flip);
  * пропускная способность (ТС/с) при батче 32;
  * то же для признака фона (bg.onnx), он нужен один раз на кадр.

    python bench.py --data ./data --weights weights
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import psutil

from infer import background_tensor, vehicle_tensor
from reid.data import read_csv


def timeit(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return np.array(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    q = read_csv(f"{args.data}/test_query.csv")
    rows = list(q.itertuples())[:64]
    img = f"{args.data}/images"
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    reid = ort.InferenceSession(f"{args.weights}/reid.onnx", opts, providers=["CPUExecutionProvider"])
    bg = ort.InferenceSession(f"{args.weights}/bg.onnx", opts, providers=["CPUExecutionProvider"])

    x1 = vehicle_tensor(img, rows[0])[None]
    xb = np.stack([vehicle_tensor(img, r) for r in rows[:32]])
    b1 = background_tensor(img, rows[0])[None]
    bb = np.stack([background_tensor(img, r) for r in rows[:32]])

    def reid_one():
        reid.run(None, {"image": x1})
        reid.run(None, {"image": np.ascontiguousarray(x1[..., ::-1])})

    def reid_batch():
        reid.run(None, {"image": xb})
        reid.run(None, {"image": np.ascontiguousarray(xb[..., ::-1])})

    pre = timeit(lambda: vehicle_tensor(img, rows[0]), args.n)
    t1 = timeit(reid_one, args.n)
    tb = timeit(reid_batch, max(5, args.n // 5))
    g1 = timeit(lambda: bg.run(None, {"image": b1}), args.n)
    gb = timeit(lambda: bg.run(None, {"image": bb}), max(5, args.n // 5))

    res = {
        "cpu": f"{psutil.cpu_count(logical=False)} ядер / {psutil.cpu_count()} потоков",
        "vehicle_ms_batch1_median": round(float(np.median(t1)) * 1000, 1),
        "vehicle_ms_batch1_p95": round(float(np.percentile(t1, 95)) * 1000, 1),
        "vehicle_preprocess_ms": round(float(np.median(pre)) * 1000, 1),
        "vehicle_fps_batch32": round(32 / float(np.median(tb)), 1),
        "background_ms_batch1_median": round(float(np.median(g1)) * 1000, 1),
        "background_fps_batch32": round(32 / float(np.median(gb)), 1),
        "weights_mb": round(sum(os.path.getsize(f"{args.weights}/{f}") for f in
                                ("reid.onnx", "bg.onnx", "cam_bank.npz")) / 1e6),
    }
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
