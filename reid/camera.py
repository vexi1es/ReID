"""
Определение камеры по фону кадра. Камеры неподвижны, поэтому уменьшенный
полный кадр с закрытым bbox ТС почти однозначно задаёт камеру.
Номер и само ТС здесь не участвуют.
"""
from pathlib import Path

import numpy as np
from PIL import Image

TW, TH = 96, 54  # 1920x1080 / 20


def thumb(images_dir, row):
    img = Image.open(Path(images_dir) / f"{row.image_id}.jpg")
    W, H = img.size
    img.draft("L", (TW * 2, TH * 2))  # быстрая JPEG-декимация
    a = np.asarray(img.convert("L").resize((TW, TH), Image.BILINEAR), dtype=np.float32)
    sx, sy = TW / W, TH / H
    x0, y0 = int(row.x * sx), int(row.y * sy)
    x1, y1 = int(np.ceil((row.x + row.w) * sx)), int(np.ceil((row.y + row.h) * sy))
    mask = np.ones_like(a)
    mask[y0:y1, x0:x1] = 0
    return a, mask


def thumbs(df, images_dir):
    out = [thumb(images_dir, r) for r in df.itertuples()]
    return np.stack([a for a, _ in out]), np.stack([m for _, m in out])


def bg_distance(a1, m1, a2, m2):
    """Попарная средняя |разница| по пикселям, открытым в обоих кадрах. (n1, n2)"""
    n1, n2 = len(a1), len(a2)
    d = np.empty((n1, n2), np.float32)
    A2, M2 = a2.reshape(n2, -1), m2.reshape(n2, -1)
    for i in range(n1):
        m = m1[i].ravel()[None] * M2
        d[i] = (np.abs(a1[i].ravel()[None] - A2) * m).sum(1) / np.clip(m.sum(1), 1, None)
    return d
