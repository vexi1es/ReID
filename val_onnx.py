"""
Точность и скорость ONNX-вариантов модели на локальной валидации (без torch):
fp32 / int8, с flip-TTA и без. Итоговая метрика — как в eval_run.py (камера + re-ranking).

    python val_onnx.py --data <dataset> --crops crops --models weights/reid_v1.onnx weights/reid_v1_int8.onnx
"""
import argparse
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from reid import official_eval as ev
from reid.data import make_val_split, read_csv
from reid.rerank import k_reciprocal

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(crops, image_id, size=224):
    img = Image.open(f"{crops}/{image_id}.jpg").convert("RGB").resize((size, size), Image.BICUBIC)
    return ((np.asarray(img, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def embed(sess, X, flip, batch=32):
    out = []
    for i in range(0, len(X), batch):
        x = X[i:i + batch]
        e = sess.run(None, {"image": x})[0]
        if flip:
            e = e + sess.run(None, {"image": np.ascontiguousarray(x[..., ::-1])})[0]
        out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    return np.concatenate(out)


def ranking(M, q, g):
    gi = g.image_id.tolist()
    order = np.argsort(-M, 1, kind="stable")[:, :10]
    rm = ev.ranking_metrics(q.set_index("image_id"), g.set_index("image_id"),
                            {a: [gi[j] for j in r] for a, r in zip(q.image_id, order)})
    return rm["mAP@10"], rm["Rank-1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--bg-cache", default="cache_bg_train.npy")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    train = read_csv(f"{args.data}/train.csv")
    rest, gt = make_val_split(train)
    q, g = gt[gt.split == "query"], gt[gt.split == "gallery"]

    F = np.load(args.bg_cache)
    ri = train.index[train.image_id.isin(rest.image_id)].values
    vi = train.image_id.reset_index().set_index("image_id").loc[gt.image_id, "index"].values
    S = F[vi] @ F[ri].T
    cr = train.camera_id.values[ri]
    top = np.argsort(-S, 1)[:, :5]
    cam = np.array([np.bincount(cr[r], weights=np.exp((S[i, r] - S[i, r].max()) / 0.02)).argmax()
                    for i, r in enumerate(top)])
    same = cam[q.index][:, None] == cam[g.index][None]

    X = np.stack([load(args.crops, i) for i in gt.image_id])
    print(f"валидация: {len(q)} query, {len(g)} gallery")
    for path in args.models:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        x1 = X[:1]
        for flip in (True, False):
            t = []
            for _ in range(20):
                t0 = time.perf_counter()
                sess.run(None, {"image": x1})
                if flip:
                    sess.run(None, {"image": np.ascontiguousarray(x1[..., ::-1])})
                t.append(time.perf_counter() - t0)
            emb = embed(sess, X, flip)
            qe, ge = emb[q.index], emb[g.index]
            rr = -k_reciprocal(qe, ge, k1=10, k2=3, lam=0.3)
            m_raw, r_raw = ranking(qe @ ge.T, q, g)
            m_fin, r_fin = ranking(np.where(same, rr - 10, rr), q, g)
            print(f"{path.split('/')[-1]:<22} flip={'да ' if flip else 'нет'} | "
                  f"итог mAP {m_fin:.4f} R1 {r_fin:.3f} (без пост-обр. {m_raw:.4f}) | "
                  f"батч1 {np.median(t) * 1000:.0f} мс", flush=True)


if __name__ == "__main__":
    main()
