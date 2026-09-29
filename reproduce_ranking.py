"""
Пересборка submission.csv и candidates.csv только из embeddings.npy и ranking_inputs.csv — без картинок
и без нейросетей. Показывает, что ранжирование получено из эмбеддингов модели, и объясняет,
чем оно отличается от простого косинуса:

  1. косинус эмбеддингов  ->  2. k-reciprocal re-ranking (Zhong, CVPR 2017)
  3. кадры галереи с той же камерой, что и запрос (camera_pred: камера по фону кадра), — вниз
  4. бонус парам противоположного ракурса (view_pred: front/rear)
  5. candidates: top-1 итогового ранжирования, уверенность — косинус, отказ ниже порога

    python reproduce_ranking.py --sub submission/ --data <папка с test_query.csv и test_gallery.csv> [--check]
"""
import argparse
import filecmp
import json
import os
import tempfile

import numpy as np
import pandas as pd

from infer import RERANK, THRESHOLD
from reid.rerank import k_reciprocal
from reid.submit import candidates_df, write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission", help="папка с embeddings.npy и ranking_inputs.csv")
    ap.add_argument("--data", required=True, help="папка с test_query.csv и test_gallery.csv")
    ap.add_argument("--weights", default="weights", help="meta.json: вес бонуса ракурса")
    ap.add_argument("--out", default=None, help="куда писать; по умолчанию временная папка")
    ap.add_argument("--check", action="store_true", help="сравнить с файлами в --sub")
    args = ap.parse_args()

    q_ids = pd.read_csv(f"{args.data}/test_query.csv").image_id.tolist()
    g_ids = pd.read_csv(f"{args.data}/test_gallery.csv").image_id.tolist()
    emb = np.load(f"{args.sub}/embeddings.npy")
    aux = pd.read_csv(f"{args.sub}/ranking_inputs.csv", dtype={"image_id": str})
    assert aux.image_id.tolist() == q_ids + g_ids, "порядок ranking_inputs.csv не совпадает с test_*.csv"
    nq = len(q_ids)
    q_emb, g_emb = emb[:nq], emb[nq:]
    cam = aux.camera_pred.values
    view = aux.view_pred.astype(str).values
    q_cam, g_cam, q_view, g_view = cam[:nq], cam[nq:], view[:nq], view[nq:]
    meta_path = f"{args.weights}/meta.json"
    beta = json.load(open(meta_path, encoding="utf-8")).get("view", {}).get("beta", 0.0) \
        if os.path.exists(meta_path) else 0.08

    cos = q_emb @ g_emb.T
    score = -k_reciprocal(q_emb, g_emb, **RERANK)
    score = np.where(q_cam[:, None] == g_cam[None], score - 10, score)
    opp = (((q_view[:, None] == "front") & (g_view[None] == "rear"))
           | ((q_view[:, None] == "rear") & (g_view[None] == "front")))
    score = score + beta * opp
    order = np.argsort(-score, axis=1, kind="stable")[:, :10]

    out = args.out or tempfile.mkdtemp(prefix="reid_repro_")
    os.makedirs(out, exist_ok=True)
    write_submission(f"{out}/submission.csv", q_ids, g_ids, order)
    candidates_df(q_ids, g_ids, cos, order, THRESHOLD).to_csv(f"{out}/candidates.csv", index=False,
                                                              lineterminator="\n")

    plain = cos.argmax(1)
    cross = np.where(q_cam[:, None] == g_cam[None], -9, cos).argmax(1)
    print(f"top-1 совпадает с косинусом по всем кадрам у {(plain == order[:, 0]).mean():.1%} запросов "
          f"(по всем — почти всегда соседний кадр той же камеры, он не засчитывается), "
          f"с косинусом по другим камерам — у {(cross == order[:, 0]).mean():.1%}; "
          f"остальное — re-ranking и поправка на ракурс")
    if args.check:
        for f in ("submission.csv", "candidates.csv"):
            same = filecmp.cmp(f"{out}/{f}", f"{args.sub}/{f}", shallow=False)
            print(f"{f}: {'совпадает байт в байт' if same else 'ОТЛИЧАЕТСЯ'}")
    print("записано в", out)


if __name__ == "__main__":
    main()
