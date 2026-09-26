"""
Пост-обработка «треклет»: усредняем эмбеддинг кадра с похожими кадрами той же (предсказанной)
камеры — это почти всегда соседние кадры той же ТС. Сравнение с базой бутстрэпом.

    python track_eval.py --data <dataset>
"""
import argparse

import numpy as np

from ens_eval import per_query_ap
from reid.data import make_val_split, read_csv
from reid.rerank import k_reciprocal

BASE = [("runs/v1", 0.7), ("runs/cnxS_dv3b", 0.3)]


def pred_cams(train, rest, gt):
    F = np.load("cache_bg_train.npy")
    ri = train.index[train.image_id.isin(rest.image_id)].values
    vi = train.image_id.reset_index().set_index("image_id").loc[gt.image_id, "index"].values
    Fr, cr = F[ri], train.camera_id.values[ri]
    S = F[vi] @ Fr.T
    top = np.argsort(-S, 1)[:, :5]
    return np.array([np.bincount(cr[r], weights=np.exp((S[i, r] - S[i, r].max()) / 0.02)).argmax()
                     for i, r in enumerate(top)])


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def smooth(emb, cam, thr, alpha, pool):
    """pool — маска, с какими кадрами разрешено усреднять (индексы в emb)."""
    S = emb @ emb.T
    W = (S > thr) & (cam[:, None] == cam[None]) & pool
    np.fill_diagonal(W, False)
    agg = (W * S) @ emb
    return l2(emb + alpha * agg)


def final_ap(qe, ge, same, q, g):
    rr = -k_reciprocal(qe, ge, k1=10, k2=3, lam=0.3)
    return per_query_ap(np.where(same, rr - 10, rr), q, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    train = read_csv(f"{args.data}/train.csv")
    rest, gt = make_val_split(train)
    qi, gi = np.where(gt.split == "query")[0], np.where(gt.split == "gallery")[0]
    q, g = gt.iloc[qi], gt.iloc[gi]
    cam = pred_cams(train, rest, gt)
    same = cam[qi][:, None] == cam[gi][None]
    emb = l2(np.concatenate([np.load(f"{r}/val_emb.npy") * np.sqrt(w) for r, w in BASE], 1))

    b_ap, b_r1 = final_ap(emb[qi], emb[gi], same, q, g)
    print(f"база: mAP {b_ap.mean():.4f} R1 {b_r1.mean():.3f}", flush=True)
    idx = np.random.default_rng(0).integers(0, len(b_ap), (2000, len(b_ap)))
    n = len(gt)
    isg = np.zeros(n, bool)
    isg[gi] = True
    pools = {"gal→gal": isg[:, None] & isg[None],          # галерея только с галереей
             "всё": np.ones((n, n), bool)}                  # + запросы с запросами и с галереей той же камеры
    for pname, pool in pools.items():
        for thr in (0.5, 0.6, 0.7, 0.8):
            for alpha in (0.5, 1.0):
                e = smooth(emb, cam, thr, alpha, pool)
                c_ap, c_r1 = final_ap(e[qi], e[gi], same, q, g)
                d = c_ap - b_ap
                lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
                print(f"{pname:8} thr {thr} α {alpha}: mAP {c_ap.mean():.4f} R1 {c_r1.mean():.3f} "
                      f"Δ {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
