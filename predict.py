"""
Инференс на тесте -> submission.csv, embeddings.npy, candidates.csv.

    python build_crops.py --data <dataset> --out crops      # один раз
    python predict.py --data <dataset> --crops crops --ckpt runs/v1/best.pt --out submit

Ранжирование (submission.csv):
  1. косинус эмбеддингов ТС;
  2. k-reciprocal re-ranking (k1=10, k2=3, lambda=0.3);
  3. кадры галереи с той же камеры, что и запрос, опускаются вниз: по протоколу
     они либо junk (та же ТС), либо заведомо другая ТС. Камера определяется по
     фону полного кадра (bbox ТС закрашен) kNN-ом по кадрам train.
Режим кандидатов (candidates.csv): top-1 по чистому косинусу и порог отказа.
"""
import argparse
import json
import os

import numpy as np
import torch

from reid.camfeat import background_features
from reid.data import read_csv
from reid.loader import extract
from reid.model import ReIDModel
from reid.rerank import k_reciprocal
from reid.submit import candidates_df, write_submission

RERANK = dict(k1=10, k2=3, lam=0.3)
THRESHOLD = 0.30  # см. README: максимум F1 на валидации с 43% open-set


def build_cam_bank(data, out):
    t = read_csv(f"{data}/train.csv")
    feats = background_features(t, f"{data}/images")
    np.savez_compressed(out, feats=feats.astype(np.float16), cams=t.camera_id.values)


def predict_cameras(feats, bank, k=5, temp=0.02):
    ref, cams = bank["feats"].astype(np.float32), bank["cams"]
    S = feats @ ref.T
    top = np.argsort(-S, 1)[:, :k]
    out = np.empty(len(feats), dtype=cams.dtype)
    for i, r in enumerate(top):
        w = np.exp((S[i, r] - S[i, r].max()) / temp)
        out[i] = np.bincount(cams[r], weights=w).argmax()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--ckpt", nargs="+", required=True, help="один чекпойнт или несколько (ансамбль)")
    ap.add_argument("--ens-weights", nargs="+", type=float, default=None,
                    help="веса моделей ансамбля (по умолчанию поровну), напр. 0.7 0.3")
    ap.add_argument("--cam-bank", default="weights/cam_bank.npz")
    ap.add_argument("--backbone", default="facebook/dinov2-base")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--out", default="submit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    q = read_csv(f"{args.data}/test_query.csv")
    g = read_csv(f"{args.data}/test_gallery.csv")
    q_ids, g_ids = q.image_id.tolist(), g.image_id.tolist()

    qs, gs = [], []
    for ckpt in args.ckpt:
        log = os.path.join(os.path.dirname(ckpt), "log.json")
        a = json.load(open(log))["args"] if os.path.exists(log) else {}
        size, backbone = a.get("size", args.size), a.get("backbone", args.backbone)
        model = ReIDModel(backbone, grad_ckpt=False).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        qs.append(extract(model, q.image_id, args.crops, size, dev, workers=0))
        gs.append(extract(model, g.image_id, args.crops, size, dev, workers=0))
        del model
        torch.cuda.empty_cache()
    # ансамбль: конкатенация L2-векторов с весами sqrt(w): косинус = взвешенное среднее косинусов
    w = np.array(args.ens_weights or [1.0] * len(qs), dtype=np.float32)
    w = w / w.sum()
    q_emb = np.concatenate([e * np.sqrt(wi) for e, wi in zip(qs, w)], 1)
    g_emb = np.concatenate([e * np.sqrt(wi) for e, wi in zip(gs, w)], 1)
    np.save(f"{args.out}/embeddings.npy", np.concatenate([q_emb, g_emb]))

    cos = q_emb @ g_emb.T
    score = -k_reciprocal(q_emb, g_emb, **RERANK) if not args.no_rerank else cos.copy()
    if not args.no_camera:
        if not os.path.exists(args.cam_bank):
            os.makedirs(os.path.dirname(args.cam_bank), exist_ok=True)
            build_cam_bank(args.data, args.cam_bank)
        bank = np.load(args.cam_bank)
        q_cam = predict_cameras(background_features(q, f"{args.data}/images"), bank)
        g_cam = predict_cameras(background_features(g, f"{args.data}/images"), bank)
        score = np.where(q_cam[:, None] == g_cam[None], score - 10, score)
    order = np.argsort(-score, axis=1, kind="stable")[:, :10]
    write_submission(f"{args.out}/submission.csv", q_ids, g_ids, order)

    # кандидат = top-1 итогового (кросс-камерного) ранжирования, уверенность — его косинус:
    # «двойник» с той же камеры не мешает отказу (TNR), см. README «Порог отказа»
    cands = candidates_df(q_ids, g_ids, cos, order, args.threshold)
    cands.to_csv(f"{args.out}/candidates.csv", index=False)
    print(f"{args.out}: {len(q)} query, {len(g)} gallery; кандидатов {len(cands)} "
          f"(отказов {len(q) - len(cands)}), порог {args.threshold:.2f}")


if __name__ == "__main__":
    main()
