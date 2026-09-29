"""
Демо-галерея для сервиса из готовых артефактов сдачи (без повторного прогона моделей):
все кадры теста — test_gallery.csv (750) и test_query.csv (1110) = 1860 ТС.
Эмбеддинги — submission/embeddings.npy, камеры — submission/ranking_inputs.csv.

    python service/build_demo_store.py --data <dataset> --sub submission --out <папка store>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "service", "api")]

from engine import crop_box  # noqa: E402
from store import Store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sub", default="submission")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    q = pd.read_csv(f"{args.data}/test_query.csv")
    g = pd.read_csv(f"{args.data}/test_gallery.csv")
    emb = np.load(f"{args.sub}/embeddings.npy")
    aux = pd.read_csv(f"{args.sub}/ranking_inputs.csv")
    rows = pd.concat([g.assign(part="test_gallery"), q.assign(part="test_query")], ignore_index=True)
    order = {k: i for i, k in enumerate(aux.image_id)}          # эмбеддинги: сначала query, затем gallery
    if os.path.exists(f"{args.out}/gallery.db"):
        os.remove(f"{args.out}/gallery.db")
    store = Store(args.out, dim=emb.shape[1])
    store.clear()
    for n, r in enumerate(rows.itertuples(), 1):
        i = order[r.image_id]
        img = Image.open(f"{args.data}/images/{r.image_id}.jpg").convert("RGB")
        bbox = (r.x, r.y, r.w, r.h)
        store.add(emb[i], crop_box(img, bbox), label=r.image_id, camera=int(aux.camera_pred[i]),
                  source=r.part, bbox=bbox)
        if n % 300 == 0:
            print(n, "/", len(rows), flush=True)
    print("готово:", store.count(), "ТС ->", args.out)


if __name__ == "__main__":
    main()
