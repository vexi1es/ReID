"""
Инференс решения одной командой, без torch и без интернета (onnxruntime: GPU, если есть, иначе CPU).

    python infer.py --data /data --out /out

/data: test_query.csv, test_gallery.csv, images/ (+ train.csv не нужен).
/out:  submission.csv, embeddings.npy, candidates.csv.

Шаги (повторяют predict.py, где считались метрики валидации):
  1. вырезаем ТС по bbox (+5% запаса) -> 224x224 -> reid.onnx, flip-TTA -> эмбеддинг;
  2. полный кадр 392x224 с закрашенным bbox -> bg.onnx -> камера (kNN по банку фонов train);
  3. ранжирование: k-reciprocal re-ranking, кадры той же камеры, что и запрос, — вниз;
     если в weights/meta.json есть блок "view": ракурс ТС (view.onnx) и бонус парам спереди<->сзади;
  4. candidates: top-1 итогового ранжирования (другие камеры), уверенность — косинус, отказ ниже порога.
"""
import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image, ImageDraw

from reid.data import crop, read_csv
from reid.rerank import k_reciprocal
from reid.submit import candidates_df, write_submission

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CROP, SIZE = 320, 224          # как в build_crops.py и при обучении
BG_W, BG_H = 392, 224          # как в reid/camfeat.py
RERANK = dict(k1=10, k2=3, lam=0.3)
THRESHOLD = 0.30


def available_cpus():
    """Сколько ядер реально выдано процессу: квота Docker (cgroup) и привязка к ядрам.
    Без этого onnxruntime в контейнере с --cpus N запускает потоков по числу ядер машины
    и в разы замедляется от борьбы за квоту."""
    n = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        n = min(n, len(os.sched_getaffinity(0)))
    try:  # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except Exception:
        try:  # cgroup v1
            quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except Exception:
            pass
    return n


def ort_providers():
    """CUDA, если установлен onnxruntime-gpu и видна видеокарта (docker-compose.gpu.yml), иначе CPU."""
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            ort.preload_dlls()          # CUDA/cuDNN из pip-пакетов nvidia-* (onnxruntime-gpu[cuda,cudnn])
        except Exception:
            pass
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _norm(img):
    return ((np.asarray(img, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def vehicle_tensor(images_dir, row, size=SIZE):
    # точь-в-точь как при обучении: кроп 320x320 -> JPEG q92 (build_crops.py) -> 224 bicubic
    buf = io.BytesIO()
    crop(images_dir, row).resize((CROP, CROP)).save(buf, "JPEG", quality=92)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((size, size), Image.BICUBIC)
    return _norm(img)


def background_tensor(images_dir, row):
    img = Image.open(Path(images_dir) / f"{row.image_id}.jpg")
    sx, sy = BG_W / img.size[0], BG_H / img.size[1]
    img.draft("RGB", (BG_W * 2, BG_H * 2))
    img = img.convert("RGB").resize((BG_W, BG_H), Image.BILINEAR)
    ImageDraw.Draw(img).rectangle(
        [row.x * sx, row.y * sy, (row.x + row.w) * sx, (row.y + row.h) * sy], fill=(124, 116, 104))
    return _norm(img)


def run(session, loader, df, images_dir, batch, pool, flip=False):
    rows = list(df.itertuples())
    out = []
    for i in range(0, len(rows), batch):
        x = np.stack(list(pool.map(lambda r: loader(images_dir, r), rows[i:i + batch])))
        e = session.run(None, {"image": x})[0]
        if flip:
            e = e + session.run(None, {"image": np.ascontiguousarray(x[..., ::-1])})[0]
            e /= np.linalg.norm(e, axis=1, keepdims=True)
        out.append(e.astype(np.float32))
    return np.concatenate(out)


def predict_cameras(feats, bank, k=5, temp=0.02):
    ref, cams = bank["feats"].astype(np.float32), bank["cams"]
    S = feats @ ref.T
    top = np.argsort(-S, 1)[:, :k]
    out = np.empty(len(feats), dtype=cams.dtype)
    for i, r in enumerate(top):
        out[i] = np.bincount(cams[r], weights=np.exp((S[i, r] - S[i, r].max()) / temp)).argmax()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--fast", action="store_true",
                    help="без flip-TTA: вдвое быстрее, итог mAP@10 на валидации 0.7125 вместо 0.7229")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    images = f"{args.data}/images"
    q = read_csv(f"{args.data}/test_query.csv")
    g = read_csv(f"{args.data}/test_gallery.csv")
    q_ids, g_ids = q.image_id.tolist(), g.image_id.tolist()

    cpus = available_cpus()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = cpus
    opts.inter_op_num_threads = 1
    prov = ort_providers()
    bg = ort.InferenceSession(f"{args.weights}/bg.onnx", opts, providers=prov)
    bank = np.load(f"{args.weights}/cam_bank.npz")
    meta_path = f"{args.weights}/meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    models = meta.get("models") or [{"onnx": "reid.onnx", "size": meta.get("size", SIZE), "weight": 1.0}]
    total = sum(m["weight"] for m in models)
    print(f"{prov[0].replace('ExecutionProvider', '')}, CPU {cpus} потоков; модели: " +
          ", ".join(f"{m['onnx']} {m['size']}px x{m['weight'] / total:.2f}" for m in models), flush=True)

    with ThreadPoolExecutor(min(8, cpus)) as pool:
        qs, gs = [], []
        for m in models:
            sess = ort.InferenceSession(f"{args.weights}/{m['onnx']}", opts, providers=prov)
            vehicle = lambda d, r, s=m["size"]: vehicle_tensor(d, r, s)
            k = np.sqrt(m["weight"] / total)   # косинус ансамбля = взвешенное среднее косинусов
            qs.append(run(sess, vehicle, q, images, args.batch, pool, flip=not args.fast) * k)
            gs.append(run(sess, vehicle, g, images, args.batch, pool, flip=not args.fast) * k)
        q_emb, g_emb = np.concatenate(qs, 1), np.concatenate(gs, 1)
        np.save(f"{args.out}/embeddings.npy", np.concatenate([q_emb, g_emb]))
        q_cam = predict_cameras(run(bg, background_tensor, q, images, args.batch, pool), bank)
        g_cam = predict_cameras(run(bg, background_tensor, g, images, args.batch, pool), bank)
        view = meta.get("view")
        if view:
            vs = ort.InferenceSession(f"{args.weights}/{view['onnx']}", opts, providers=prov)
            vt = lambda d, r, s=view["size"]: vehicle_tensor(d, r, s)
            cls = np.array(view["classes"])
            q_view = cls[run(vs, vt, q, images, args.batch, pool).argmax(1)]
            g_view = cls[run(vs, vt, g, images, args.batch, pool).argmax(1)]

    cos = q_emb @ g_emb.T
    score = -k_reciprocal(q_emb, g_emb, **RERANK)
    score = np.where(q_cam[:, None] == g_cam[None], score - 10, score)
    if view:
        # тот же ТС с противоположной стороны похож меньше, чем соседний ТС с той же стороны
        # (VANet, ICCV 2019; VOC-ReID, AI City 2020): бонус парам спереди<->сзади, +0.021 mAP на валидации
        opp = (((q_view[:, None] == "front") & (g_view[None] == "rear"))
               | ((q_view[:, None] == "rear") & (g_view[None] == "front")))
        score = score + view["beta"] * opp
    else:
        q_view, g_view = np.full(len(q), "-"), np.full(len(g), "-")
    order = np.argsort(-score, axis=1, kind="stable")[:, :10]
    write_submission(f"{args.out}/submission.csv", q_ids, g_ids, order)
    # кандидат = top-1 итогового кросс-камерного ранжирования, уверенность — его косинус
    cands = candidates_df(q_ids, g_ids, cos, order, args.threshold)
    cands.to_csv(f"{args.out}/candidates.csv", index=False, lineterminator="\n")
    # всё, кроме эмбеддингов, что нужно для ранжирования: по этому файлу и embeddings.npy
    # reproduce_ranking.py пересобирает submission.csv и candidates.csv байт в байт
    pd.DataFrame({"image_id": q_ids + g_ids, "part": ["query"] * len(q) + ["gallery"] * len(g),
                  "camera_pred": np.concatenate([q_cam, g_cam]),
                  "view_pred": np.concatenate([q_view, g_view])}).to_csv(
        f"{args.out}/ranking_inputs.csv", index=False, lineterminator="\n")
    print(f"{args.out}: {len(q)} query, {len(g)} gallery, кандидатов {len(cands)}, "
          f"{time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
