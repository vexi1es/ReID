"""
Проверка идей постобработки из исследования (27.09) на валидации «как тест», без переобучения:
  A. сетка k-reciprocal (k1, k2, λ);
  B. вычитание среднего вектора по предсказанной камере (DMT, AI City 2021);
  C. сложение расстояний моделей после re-ranking каждой вместо склейки векторов;
  D. бонус к парам противоположного ракурса (VOC-ReID / VABPP).
Каждый вариант — парный бутстрэп против текущего финала. Чтобы не подогнать параметры под
валидацию, лучшие настройки подбираются на половине запросов и проверяются на другой.

    python postproc2_eval.py --data <dataset>
"""
import argparse
import itertools
import json

import numpy as np

from ens_eval import per_query_ap
from reid.data import make_val_split, read_csv
from reid.rerank import k_reciprocal
from track_eval import l2, pred_cams

MODELS = [("runs/v1", 0.7), ("runs/cnxS_dv3b", 0.3)]
BASE_RR = dict(k1=10, k2=3, lam=0.3)
OPP = {("front", "rear"), ("rear", "front")}


def rr_score(qe, ge, same, **rr):
    s = -k_reciprocal(qe, ge, **rr)
    return np.where(same, s - 10, s)


def cam_center(emb, cam, a):
    out = emb.copy()
    for c in np.unique(cam):
        m = cam == c
        out[m] -= a * emb[m].mean(0)
    return l2(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/postproc2.json")
    args = ap.parse_args()

    train = read_csv(f"{args.data}/train.csv")
    rest, gt = make_val_split(train)
    qi, gi = np.where(gt.split == "query")[0], np.where(gt.split == "gallery")[0]
    q, g = gt.iloc[qi], gt.iloc[gi]
    cam = pred_cams(train, rest, gt)
    same = cam[qi][:, None] == cam[gi][None]
    views = np.load("runs/val_views.npy", allow_pickle=True)
    assert len(views) == len(gt)
    opp = np.array([[(a, b) in OPP for b in views[gi]] for a in views[qi]])

    parts = [l2(np.load(f"{r}/val_emb.npy")) for r, _ in MODELS]
    emb = l2(np.concatenate([p * np.sqrt(w) for p, (_, w) in zip(parts, MODELS)], 1))

    def ev(score):
        return per_query_ap(score, q, g)

    b_ap, b_r1 = ev(rr_score(emb[qi], emb[gi], same, **BASE_RR))
    n = len(b_ap)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, n, (2000, n))
    half = rng.permutation(n)
    A_idx, B_idx = half[: n // 2], half[n // 2:]
    print(f"база (финал): mAP {b_ap.mean():.4f} R1 {b_r1.mean():.3f}  [{n} запросов]", flush=True)
    res = {"base": {"mAP": b_ap.mean(), "R1": b_r1.mean()}, "exp": []}

    def report(name, ap_, r1_, params):
        d = ap_ - b_ap
        lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
        row = dict(name=name, params=params, mAP=ap_.mean(), R1=r1_.mean(), d=d.mean(), ci=[lo, hi],
                   dA=d[A_idx].mean(), dB=d[B_idx].mean())
        res["exp"].append(row)
        return row

    def sweep(title, grid, make):
        """make(params) -> (ap, r1). Лучший на половине A, проверка на B и на всех."""
        rows = []
        for p in grid:
            ap_, r1_ = make(p)
            rows.append(report(title, ap_, r1_, p))
        best = max(rows, key=lambda r: r["dA"])
        top = sorted(rows, key=lambda r: -r["d"])[:3]
        print(f"\n== {title}: лучший по половине A {best['params']} -> на B {best['dB']:+.4f}, "
              f"на всех {best['d']:+.4f} CI [{best['ci'][0]:+.4f}, {best['ci'][1]:+.4f}] R1 {best['R1']:.3f}")
        for r in top:
            print(f"   {r['params']}: Δ {r['d']:+.4f} CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] R1 {r['R1']:.3f}")
        return best

    # A. сетка re-ranking
    grid = [dict(k1=k1, k2=k2, lam=lam) for k1, k2, lam in
            itertools.product([4, 6, 7, 8, 10, 14], [1, 2, 3], [0.2, 0.3, 0.45, 0.6])]
    bestA = sweep("A re-ranking", grid, lambda p: ev(rr_score(emb[qi], emb[gi], same, **p)))

    # B. вычитание среднего по камере (до склейки, по каждой модели)
    def camsub(a):
        e = l2(np.concatenate([cam_center(p, cam, a) * np.sqrt(w) for p, (_, w) in zip(parts, MODELS)], 1))
        return e
    bestB = sweep("B среднее по камере", [dict(a=a) for a in (0.05, 0.1, 0.18, 0.25, 0.35, 0.5)],
                  lambda p: ev(rr_score(camsub(p["a"])[qi], camsub(p["a"])[gi], same, **BASE_RR)))

    # C. сложение расстояний после re-ranking каждой модели
    per = [-k_reciprocal(p[qi], p[gi], **BASE_RR) for p in parts]
    bestC = sweep("C сумма расстояний", [dict(w=w) for w in (0.5, 0.6, 0.7, 0.8)],
                  lambda p: ev(np.where(same, p["w"] * per[0] + (1 - p["w"]) * per[1] - 10,
                                        p["w"] * per[0] + (1 - p["w"]) * per[1])))

    # D. бонус противоположному ракурсу
    base_s = rr_score(emb[qi], emb[gi], same, **BASE_RR)
    bestD = sweep("D бонус ракурса", [dict(beta=b) for b in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)],
                  lambda p: ev(base_s + p["beta"] * opp))

    # E. комбинация лучших (параметры с половины A)
    def combo():
        a = bestB["params"]["a"] if bestB["dA"] > 0 else 0.0
        e = camsub(a) if a else emb
        rr = bestA["params"] if bestA["dA"] > 0 else BASE_RR
        s = rr_score(e[qi], e[gi], same, **rr)
        beta = bestD["params"]["beta"] if bestD["dA"] > 0 else 0.0
        return ev(s + beta * opp), dict(a=a, rr=rr, beta=beta)
    (ap_, r1_), p = combo()
    r = report("E комбинация", ap_, r1_, p)
    print(f"\n== E комбинация {p}: Δ {r['d']:+.4f} CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] "
          f"(A {r['dA']:+.4f}, B {r['dB']:+.4f}) mAP {r['mAP']:.4f} R1 {r['R1']:.3f}")
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)


if __name__ == "__main__":
    main()
