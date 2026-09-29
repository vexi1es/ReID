"""
Детектор ТС для веб-интерфейса (автоопределение рамки на загруженном кадре). В задачу хакатона
детекция не входит (bbox даются организаторами) — это удобство для оператора.

torchvision Faster R-CNN MobileNetV3-Large 320 FPN, веса COCO (лицензия BSD-3), экспорт в ONNX.
Классы ТС в COCO: 3 car, 4 motorcycle, 6 bus, 8 truck.

    python export_detector.py --out weights            # -> weights/detector.onnx
    python export_detector.py --check <dataset>        # IoU с разметкой на кадрах теста
"""
import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torchvision

VEHICLE = {3: "легковой", 4: "мотоцикл", 6: "автобус", 8: "грузовик"}


class Detector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights="DEFAULT", box_score_thresh=0.3)

    def forward(self, image):                    # image: [3, H, W], 0..1
        out = self.m([image])[0]
        return out["boxes"], out["labels"], out["scores"]


def export(path, sample=None):
    net = Detector().eval()
    if sample:                                   # трассировка на реальном кадре с машинами
        from PIL import Image
        x = torch.from_numpy(_letterbox(Image.open(sample).convert("RGB"))[0])
    else:
        x = torch.rand(3, 540, 960)
    torch.onnx.export(net, (x,), path, input_names=["image"], output_names=["boxes", "labels", "scores"],
                      dynamic_axes={"boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
                      opset_version=17, dynamo=False)
    print(path, os.path.getsize(path) // 2**20, "МБ")


def _letterbox(img, W0=960, H0=540):
    """Кадр любого размера -> 3x540x960 (масштаб с сохранением пропорций + поля)."""
    W, H = img.size
    s = min(W0 / W, H0 / H)
    small = img.resize((max(1, int(W * s)), max(1, int(H * s))))
    canvas = np.zeros((H0, W0, 3), np.float32)
    canvas[:small.size[1], :small.size[0]] = np.asarray(small, np.float32) / 255
    return canvas.transpose(2, 0, 1).copy(), s


def detect(sess, img, min_score=0.4):
    """img: PIL RGB -> список (x, y, w, h, score, класс) в пикселях исходного кадра."""
    x, s = _letterbox(img)
    boxes, labels, scores = sess.run(None, {"image": x})
    out = []
    for b, l, sc in zip(boxes / s, labels, scores):
        if int(l) in VEHICLE and sc >= min_score:
            x0, y0, x1, y1 = [float(v) for v in b]
            out.append((x0, y0, x1 - x0, y1 - y0, float(sc), VEHICLE[int(l)]))
    return sorted(out, key=lambda r: -r[2] * r[3])


def check(path, data, n=150):
    import pandas as pd
    from PIL import Image
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    q = pd.read_csv(f"{data}/test_query.csv").sample(n, random_state=0)
    ious, found = [], 0
    import time
    t0 = time.time()
    for r in q.itertuples():
        img = Image.open(f"{data}/images/{r.image_id}.jpg").convert("RGB")
        dets = detect(sess, img)
        gx0, gy0, gx1, gy1 = r.x, r.y, r.x + r.w, r.y + r.h
        best = 0.0
        for x, y, w, h, *_ in dets:
            ix = max(0, min(gx1, x + w) - max(gx0, x)); iy = max(0, min(gy1, y + h) - max(gy0, y))
            inter = ix * iy
            best = max(best, inter / (r.w * r.h + w * h - inter))
        ious.append(best)
        found += bool(dets) and best >= 0.5
    print(f"{n} кадров: разметка найдена (IoU>=0.5) в {found / n:.1%}, средний лучший IoU {np.mean(ious):.3f}, "
          f"{(time.time() - t0) / n * 1000:.0f} мс/кадр")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weights")
    ap.add_argument("--check", default=None)
    ap.add_argument("--sample", default=None, help="кадр с машинами для трассировки экспорта")
    a = ap.parse_args()
    path = f"{a.out}/detector.onnx"
    if not os.path.exists(path):
        export(path, a.sample)
    if a.check:
        check(path, a.check)
