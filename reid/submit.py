"""Ранжирование и запись артефактов сдачи: submission.csv, embeddings.npy, candidates.csv."""
import csv

import numpy as np
import pandas as pd

TOP_K = 10


def rank(q_emb, g_emb, top_k=TOP_K):
    sims = q_emb @ g_emb.T
    order = np.argsort(-sims, axis=1, kind="stable")[:, :top_k]
    return sims, order


def write_submission(path, q_ids, g_ids, order):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for qid, row in zip(q_ids, order):
            w.writerow([qid, *[g_ids[j] for j in row]])


def candidates_df(q_ids, g_ids, sims, order, threshold):
    """Top-1 из order (итоговое ранжирование), уверенность — sims; ниже порога — отказ (строк нет)."""
    rows = []
    for i, qid in enumerate(q_ids):
        j = order[i, 0]
        if sims[i, j] >= threshold:
            rows.append((qid, g_ids[j], round(float(sims[i, j]), 4)))
    return pd.DataFrame(rows, columns=["query_id", "gallery_id", "confidence"])


def write_all(out_dir, q_ids, g_ids, q_emb, g_emb, threshold):
    sims, order = rank(q_emb, g_emb)
    write_submission(f"{out_dir}/submission.csv", q_ids, g_ids, order)
    np.save(f"{out_dir}/embeddings.npy", np.concatenate([q_emb, g_emb]))
    candidates_df(q_ids, g_ids, sims, order, threshold).to_csv(
        f"{out_dir}/candidates.csv", index=False)
