"""
Один раз вырезает все ТС (bbox + запас) и сохраняет квадратными CROP×CROP JPEG.
Обучение и инференс дальше читают только кеш — полные кадры 1920x1080 больше не декодируются.

    python build_crops.py --data <папка dataset> --out crops
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import psutil

from reid.data import crop, read_csv

CROP = 320


def _work(args):
    images_dir, out, rows = args
    for r in rows.itertuples():
        path = f"{out}/{r.image_id}.jpg"
        if not os.path.exists(path):
            crop(images_dir, r).resize((CROP, CROP)).save(path, quality=92)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="crops")
    ap.add_argument("--workers", type=int, default=max(1, int(os.cpu_count() * 0.9)))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # ~90% ресурсов: фоновый приоритет, ноутбук остаётся отзывчивым.
    # Потоки, а не процессы: PIL декодирует без GIL, а память — одна копия.
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if psutil.WINDOWS else 10)

    df = pd.concat([read_csv(f"{args.data}/{n}.csv") for n in ("train", "test_query", "test_gallery")])
    df = df.drop_duplicates("image_id")
    chunks = [(f"{args.data}/images", args.out, df.iloc[i::args.workers]) for i in range(args.workers)]
    with ThreadPoolExecutor(args.workers) as ex:
        print("crops:", sum(ex.map(_work, chunks)))


if __name__ == "__main__":
    main()
