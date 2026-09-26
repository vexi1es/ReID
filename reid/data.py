"""Чтение разметки и вырезание ТС по bbox."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PAD = 0.05  # запас вокруг bbox: не режем кузов впритык


def read_csv(path):
    return pd.read_csv(path, dtype={"image_id": str})


def available(df, images_dir):
    """Оставляет строки, для которых картинка есть на диске (dev-режим на неполных данных)."""
    have = {p.stem for p in Path(images_dir).glob("*.jpg")}
    return df[df.image_id.isin(have)].reset_index(drop=True)


def crop(images_dir, row, pad=PAD):
    img = Image.open(Path(images_dir) / f"{row.image_id}.jpg").convert("RGB")
    W, H = img.size
    px, py = row.w * pad, row.h * pad
    box = (
        int(max(0, row.x - px)),
        int(max(0, row.y - py)),
        int(min(W, row.x + row.w + px)),
        int(min(H, row.y + row.h + py)),
    )
    return img.crop(box)


def make_val_split(train, frac=0.2, p_query=0.6, p_no_gallery=0.2, seed=0, mode="testlike"):
    """
    Локальная валидация по протоколу теста из отложенных ТС train.
    Набор отложенных ТС зависит только от seed и frac — одинаков во всех режимах.

    mode="testlike" (по умолчанию): каждая группа (ТС, камера) делится между
      query и gallery. Так совпадает статистика теста: q/g ~1.4 и ~52% запросов
      с почти тем же bbox в галерее (junk-двойник). Open-set при этом редок (~1-2%).
    mode="openset": кадры случайно делятся на query/gallery, у доли p_no_gallery
      ТС галерея пустая — много open-set, для проверки режима отказа.
    Возвращает (train_rest, gt) — gt в формате ground truth evaluate.py.
    """
    rng = np.random.default_rng(seed)
    vids = train.vehicle_id.unique()
    hold = set(rng.choice(vids, int(len(vids) * frac), replace=False))
    rest = train[~train.vehicle_id.isin(hold)].reset_index(drop=True)
    val = train[train.vehicle_id.isin(hold)].reset_index(drop=True)

    if mode == "openset":
        no_gal = {v for v in hold if rng.random() < p_no_gallery}
        split = np.where(rng.random(len(val)) < p_query, "query", "gallery")
        split[val.vehicle_id.isin(no_gal).values] = "query"
    elif mode == "testlike":
        split = np.empty(len(val), dtype=object)
        for idx in val.groupby(["vehicle_id", "camera_id"]).indices.values():
            if len(idx) == 1:
                split[idx] = "query" if rng.random() < 0.6 else "gallery"
                continue
            lab = np.where(rng.random(len(idx)) < 0.7, "query", "gallery")
            if (lab == "query").all():
                lab[rng.integers(len(idx))] = "gallery"
            if (lab == "gallery").all():
                lab[rng.integers(len(idx))] = "query"
            split[idx] = lab
    else:
        raise ValueError(mode)
    val["split"] = split
    return rest, val
