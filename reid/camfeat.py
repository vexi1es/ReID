"""
Признаки фона кадра для определения камеры: DINOv2-small по полному кадру,
bbox ТС закрашен серым (сама машина и номер в признак не попадают).
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Dinov2Model

W, H = 392, 224  # 16:9, кратно патчу 14
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load(images_dir, row):
    img = Image.open(Path(images_dir) / f"{row.image_id}.jpg")
    sx, sy = W / img.size[0], H / img.size[1]
    img.draft("RGB", (W * 2, H * 2))
    img = img.convert("RGB").resize((W, H), Image.BILINEAR)
    ImageDraw.Draw(img).rectangle(
        [row.x * sx, row.y * sy, (row.x + row.w) * sx, (row.y + row.h) * sy], fill=(124, 116, 104))
    return ((np.asarray(img, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


@torch.no_grad()
def background_features(df, images_dir, batch=32, threads=8, name="facebook/dinov2-small"):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Dinov2Model.from_pretrained(name).to(dev).eval()
    rows = list(df.itertuples())
    out = []
    with ThreadPoolExecutor(threads) as ex:
        for i in range(0, len(rows), batch):
            x = torch.from_numpy(np.stack(list(ex.map(lambda r: _load(images_dir, r), rows[i:i + batch]))))
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                h = model(pixel_values=x.to(dev)).last_hidden_state.float()
            f = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
            print(f"\r  bg {min(i + batch, len(rows))}/{len(rows)}", end="", flush=True)
    print()
    return np.concatenate(out)
