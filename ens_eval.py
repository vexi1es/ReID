"""
Взвешенные ансамбли по кешированным val_emb.npy + парный бутстрэп против базового ансамбля.

    python ens_eval.py --data <dataset> --base runs/v1:0.7 runs/cnxS_dv3b:0.3 \
        --cand "runs/v1:0.7 runs/kg_cnxB_256:0.3" "runs/v1:0.6 runs/cnxS_dv3b:0.2 runs/kg_cnxB_256:0.2"
Итоговая метрика — как в eval_run (re-ranking + демоут камеры), AP@10 по правилам evaluate.py.
"""
import argparse
import json

import numpy as np

from eval_run import camera_same
from reid.data import make_val_split, read_csv
from reid.rerank import k_reciprocal


def parse(spec):
    out = []
    for t in spec.split() if isinstance(spec, str) else spec:
        r, w = t.rsplit(":", 1)
        out.append((r, float(w)))
    return out


def per_query_ap(M, q, g, top_k=10):
    """AP@10 и hit@1 по каждому запросу (junk = та же ТС и камера — выкидывается)."""
    qv, qc = q.vehicle_id.values, q.camera_id.values
    gv, gc = g.vehicle_id.values, g.camera_id.values
    aps, r1 = [], []
    for i in range(len(q)):
        junk = (gv == qv[i]) & (gc == qc[i])
        n_pos = int(((gv == qv[i]) & ~junk).sum())
        if n_pos == 0:
            continue
        order = np.argsort(-M[i], kind="stable")[:top_k]
        order = order[~junk[order]]
        rel = gv[order] == qv[i]
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append((prec * rel).sum() / min(n_pos, top_k))
        r1.append(bool(rel[:1].any()))
    return np.array(aps), np.array(r1)


def score(members, q, g, same):
    emb = np.concatenate([np.load(f"{r}/val_emb.npy") * np.sqrt(w) for r, w in members], 1)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    qe, ge = emb[q.index], emb[g.index]
    rr = -k_reciprocal(qe, ge, k1=10, k2=3, lam=0.3)
    return per_query_ap(np.where(same, rr - 10, rr), q, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--out", default="runs/ens_eval.json")
    args = ap.parse_args()

    train = read_csv(f"{args.data}/train.csv")
    rest, gt = make_val_split(train)
    q, g = gt[gt.split == "query"], gt[gt.split == "gallery"]
    same = camera_same(train, rest, gt, q, g)

    b_ap, b_r1 = score(parse(args.base), q, g, same)
    print(f"база {' '.join(args.base)}: mAP {b_ap.mean():.4f} R1 {b_r1.mean():.3f}", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(b_ap), (2000, len(b_ap)))
    res = {"base": {"spec": " ".join(args.base), "mAP": b_ap.mean(), "R1": b_r1.mean()}, "cand": []}
    for c in args.cand:
        c_ap, c_r1 = score(parse(c), q, g, same)
        d = c_ap - b_ap
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        print(f"{c}: mAP {c_ap.mean():.4f} R1 {c_r1.mean():.3f}  Δ {d.mean():+.4f} CI [{lo:+.4f}, {hi:+.4f}]", flush=True)
        res["cand"].append({"spec": c, "mAP": c_ap.mean(), "R1": c_r1.mean(), "d": d.mean(), "ci": [lo, hi]})
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    main()
