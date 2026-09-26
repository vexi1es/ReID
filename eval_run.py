"""
Итоговая оценка модели (или ансамбля) на валидации «как тест»: косинус, + камера, + re-ranking.
Эмбеддинги валидации кешируются в <run>/val_emb.npy.

    python eval_run.py --data <dataset> --runs runs/v1 [runs/exp2 ...] --out eval.json
"""
import argparse
import json
import os

import numpy as np
import torch

from reid import official_eval as ev
from reid.data import make_val_split, read_csv
from reid.loader import extract
from reid.model import ReIDModel
from reid.rerank import k_reciprocal


def run_args(run):
    log = json.load(open(f"{run}/log.json"))
    return log["args"]


def val_embeddings(run, gt, crops):
    cache = f"{run}/val_emb.npy"
    if os.path.exists(cache):
        return np.load(cache)
    a = run_args(run)
    ckpt = f"{run}/best.pt" if os.path.exists(f"{run}/best.pt") else f"{run}/last.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ReIDModel(a.get("backbone", "facebook/dinov2-base"), grad_ckpt=False).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location=dev))
    emb = extract(m, gt.image_id, crops, a.get("size", 224), dev, workers=0)
    np.save(cache, emb)
    del m
    torch.cuda.empty_cache()
    return emb


def camera_same(train, rest, gt, q, g):
    F = np.load("cache_bg_train.npy")
    ri = train.index[train.image_id.isin(rest.image_id)].values
    vi = train.image_id.reset_index().set_index("image_id").loc[gt.image_id, "index"].values
    Fr, cr = F[ri], train.camera_id.values[ri]
    S = F[vi] @ Fr.T
    top = np.argsort(-S, 1)[:, :5]
    cam = np.array([np.bincount(cr[r], weights=np.exp((S[i, r] - S[i, r].max()) / 0.02)).argmax()
                    for i, r in enumerate(top)])
    return cam[q.index][:, None] == cam[g.index][None]


def ranking(M, q, g):
    gi = g.image_id.tolist()
    order = np.argsort(-M, 1, kind="stable")[:, :10]
    rm = ev.ranking_metrics(q.set_index("image_id"), g.set_index("image_id"),
                            {a: [gi[j] for j in r] for a, r in zip(q.image_id, order)})
    return {"mAP": rm["mAP@10"], "R1": rm["Rank-1"], "R5": rm["Rank-5"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train = read_csv(f"{args.data}/train.csv")
    rest, gt = make_val_split(train)
    q, g = gt[gt.split == "query"], gt[gt.split == "gallery"]

    embs = [val_embeddings(r, gt, args.crops) for r in args.runs]
    emb = np.concatenate(embs, 1) / np.sqrt(len(embs))      # ансамбль: конкатенация L2-векторов
    qe, ge = emb[q.index], emb[g.index]
    cos = qe @ ge.T
    same = camera_same(train, rest, gt, q, g)
    rr = -k_reciprocal(qe, ge, k1=10, k2=3, lam=0.3)

    res = {"runs": args.runs,
           "raw": ranking(cos, q, g),
           "cam": ranking(np.where(same, cos - 2, cos), q, g),
           "final": ranking(np.where(same, rr - 10, rr), q, g)}
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
