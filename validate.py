"""
Локальная валидация: отложенные ТС из train, метрики считаются функциями
официального evaluate.py (тот же junk-фильтр, open-set, AP@10).

    python validate.py --data <папка dataset> [--dev]
"""
import argparse

import numpy as np

from reid import official_eval as ev
from reid.data import available, make_val_split, read_csv
from reid.embed import Embedder
from reid.submit import candidates_df, rank


def score(gt, emb, thresholds=np.linspace(0.3, 0.95, 66)):
    q = gt[gt.split == "query"]
    g = gt[gt.split == "gallery"]
    q_emb, g_emb = emb[q.index.values], emb[g.index.values]
    q_ids, g_ids = q.image_id.tolist(), g.image_id.tolist()
    sims, order = rank(q_emb, g_emb)

    query, gallery = q.set_index("image_id"), g.set_index("image_id")
    ranked = {qid: [g_ids[j] for j in row] for qid, row in zip(q_ids, order)}
    rm = ev.ranking_metrics(query, gallery, ranked)

    best = None
    for t in thresholds:
        c = candidates_df(q_ids, g_ids, sims, order, t)
        cands = {k: list(zip(v.gallery_id, v.confidence)) for k, v in c.groupby("query_id")}
        cm = ev.candidate_metrics(query, gallery, cands)
        if best is None or cm["F1"] > best[1]["F1"]:
            best = (t, cm)
    return rm, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dev", action="store_true", help="только картинки, что есть на диске")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = f"{args.data}/images"
    train = read_csv(f"{args.data}/train.csv")
    if args.dev:
        train = available(train, images)
    _, gt = make_val_split(train, frac=args.frac, seed=args.seed)
    q = gt[gt.split == "query"]
    print(f"val: {len(q)} query, {len(gt) - len(q)} gallery, {gt.vehicle_id.nunique()} ТС")

    emb = Embedder()(gt, images)
    rm, (t, cm) = score(gt, emb)
    print(f"mAP@10 {rm['mAP@10']:.4f}  R1 {rm['Rank-1']:.4f}  R5 {rm['Rank-5']:.4f}  "
          f"(в зачёте {rm['n_scored']}, open-set {rm['n_openset_excluded']})")
    print(f"отказ: порог {t:.2f}  F1 {cm['F1']:.4f}  P {cm['Precision']:.4f}  "
          f"R {cm['Recall']:.4f}  TNR {cm['TNR']:.4f}  PR-AUC {cm['PR-AUC']:.4f}")


if __name__ == "__main__":
    main()
