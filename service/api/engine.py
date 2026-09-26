"""
Движок признаков для сервиса: те же ONNX-модели и та же предобработка, что в infer.py
(на валидации совпадает с torch до косинуса 0.99997), но для одного кадра из памяти.

GPU используется, если установлен onnxruntime-gpu и доступна CUDA; иначе CPU.
"""
import io
import json
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from infer import BG_H, BG_W, CROP, THRESHOLD, _norm, available_cpus, ort_providers, predict_cameras
from reid.data import PAD


def crop_box(img, bbox, pad=PAD):
    """bbox = (x, y, w, h) в пикселях кадра; запас 5% — как reid.data.crop."""
    x, y, w, h = bbox
    W, H = img.size
    return img.crop((int(max(0, x - w * pad)), int(max(0, y - h * pad)),
                     int(min(W, x + w + w * pad)), int(min(H, y + h + h * pad))))


def validate_bbox(img, bbox):
    x, y, w, h = bbox
    W, H = img.size
    if w < 8 or h < 8:
        raise ValueError("bbox слишком маленький (меньше 8 px)")
    if x < 0 or y < 0 or x + w > W + 1 or y + h > H + 1:
        raise ValueError(f"bbox выходит за кадр {W}x{H}")


class Engine:
    def __init__(self, weights="weights"):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = available_cpus()
        self.providers = ort_providers()
        meta = json.load(open(f"{weights}/meta.json"))
        total = sum(m["weight"] for m in meta["models"])
        self.models = [(ort.InferenceSession(f"{weights}/{m['onnx']}", opts, providers=self.providers),
                        m["size"], np.sqrt(m["weight"] / total), m["onnx"]) for m in meta["models"]]
        self.bg = ort.InferenceSession(f"{weights}/bg.onnx", opts, providers=self.providers)
        self.bank = dict(np.load(f"{weights}/cam_bank.npz"))
        self.threshold = float(os.environ.get("THRESHOLD", THRESHOLD))

    def _vehicle(self, crop, size):
        buf = io.BytesIO()                      # JPEG q92 — как build_crops.py при обучении
        crop.resize((CROP, CROP)).save(buf, "JPEG", quality=92)
        buf.seek(0)
        return _norm(Image.open(buf).convert("RGB").resize((size, size), Image.BICUBIC))

    def embed_crops(self, crops, flip=True):
        """Эмбеддинги (L2, ансамбль) для списка уже вырезанных ТС — одним батчем на модель."""
        parts = []
        for sess, size, k, _ in self.models:
            x = np.stack([self._vehicle(c, size) for c in crops])
            e = sess.run(None, {"image": x})[0]
            if flip:
                e = e + sess.run(None, {"image": np.ascontiguousarray(x[..., ::-1])})[0]
            parts.append(e / np.linalg.norm(e, axis=1, keepdims=True) * k)
        return np.concatenate(parts, 1).astype(np.float32)

    def embed(self, img, bbox, flip=True):
        """Эмбеддинг одного ТС и его кроп для превью."""
        img = img.convert("RGB")
        validate_bbox(img, bbox)
        crop = crop_box(img, bbox)
        return self.embed_crops([crop], flip)[0], crop

    @staticmethod
    def background(img, bbox):
        x, y, w, h = bbox
        im = img.convert("RGB")
        sx, sy = BG_W / im.size[0], BG_H / im.size[1]
        im = im.resize((BG_W, BG_H), Image.BILINEAR)
        ImageDraw.Draw(im).rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], fill=(124, 116, 104))
        return _norm(im)

    def cameras(self, backgrounds):
        f = self.bg.run(None, {"image": np.stack(backgrounds)})[0].astype(np.float32)
        return [int(c) for c in predict_cameras(f, self.bank)]

    def camera(self, img, bbox):
        """Камера по фону кадра (bbox закрашен) — только для камер, известных по train."""
        return self.cameras([self.background(img, bbox)])[0]
