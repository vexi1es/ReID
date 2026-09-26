"""
Эксперименты с пост-обработкой на локальной валидации (эмбеддинги уже посчитаны).

    python postproc_eval.py --data <dataset> --ckpt runs/v1/best.pt
"""
import argparse
import os

import numpy as np
import torch

from reid import official_eval as ev
from reid.data import make_val_split, read_csv
from reid.loader import extract
from reid.model import ReIDModel


def near_dup(qb, gb, px=10):
    """Почти одинаковый bbox -> почти наверняка та же камера (на train 97%)."""
    return np.abs(qb[:, None, :] - gb[None, :, :]).max(-1) <= px


def evaluate(sims, q, g, thresholds=np.linspace(0.0, 0.95, 96)):
    q_ids, g_ids = q.image_id.tolist(), g.image_id.tolist()
    order = np.argsort(-sims, axis=1, kind="stable")
    query, gallery = q.set_index("image_id"), g.set_index("image_id")
    ranked = {qid: [g_ids[j] for j in row[:10]] for qid, row in zip(q_ids, order)}
    rm = ev.ranking_metrics(query, gallery, ranked)
    best = None
    top = sims[np.arange(len(q)), order[:, 0]]
    for t in thresholds:
        cands = {qid: [(g_ids[order[i, 0]], float(top[i]))] for i, qid in enumerate(q_ids) if top[i] >= t}
        cm = ev.candidate_metrics(query, gallery, cands)
        if best is None or cm["F1"] > best[1]["F1"]:
            best = (t, cm)
    t, cm = best
    return (f"mAP@10 {rm['mAP@10']:.4f}  R1 {rm['Rank-1']:.4f}  R5 {rm['Rank-5']:.4f} | "
            f"F1 {cm['F1']:.4f} (thr {t:.2f}, P {cm['Precision']:.3f} R {cm['Recall']:.3f} TNR {cm['TNR']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    train = read_csv(f"{args.data}/train.csv")
    _, gt = make_val_split(train)
    cache = args.ckpt.replace(".pt", "_val_emb.npy")
    if os.path.exists(cache):
        emb = np.load(cache)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = ReIDModel(grad_ckpt=False).to(dev)
        m.load_state_dict(torch.load(args.ckpt, map_location=dev))
        emb = extract(m, gt.image_id, args.crops, 224, dev, workers=0)
        np.save(cache, emb)

    for mode in ("testlike", "openset"):
        print(f"=== валидация: {mode}")
        report(make_val_split(train, mode=mode)[1], emb)


def report(gt, emb):
    q, g = gt[gt.split == "query"], gt[gt.split == "gallery"]
    S = emb[q.index] @ emb[g.index].T
    same_cam = q.camera_id.values[:, None] == g.camera_id.values[None]
    dup = near_dup(q[["x", "y", "w", "h"]].values, g[["x", "y", "w", "h"]].values)
    print(f"near-dup пары: {dup.sum()}, из них та же камера {same_cam[dup].mean():.3f}; "
          f"покрывают {dup[same_cam].mean():.3f} всех same-camera пар")

    print("baseline               ", evaluate(S, q, g))
    print("demote same-cam (oracle)", evaluate(np.where(same_cam, S - 2, S), q, g))
    print("demote near-dup bbox    ", evaluate(np.where(dup, S - 2, S), q, g))


if __name__ == "__main__":
    main()
