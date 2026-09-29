"""
Движок признаков для сервиса: те же ONNX-модели и та же предобработка, что в infer.py
(на валидации совпадает с torch до косинуса 0.99997), но для одного кадра из памяти.

GPU используется, если установлен onnxruntime-gpu и доступна CUDA; иначе CPU.
"""
import base64
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
        # reid_explain.onnx — тот же ViT, что reid.onnx (косинус 1.000000), плюс выход «вклады патчей»
        # для карты внимания; если он есть, берём его вместо reid.onnx, чтобы не держать ViT дважды
        self.explainer = None
        self.models = []
        for m in meta["models"]:
            name = m["onnx"]
            if name == "reid.onnx" and os.path.exists(f"{weights}/reid_explain.onnx"):
                name = "reid_explain.onnx"
            sess = ort.InferenceSession(f"{weights}/{name}", opts, providers=self.providers)
            if name == "reid_explain.onnx":
                self.explainer = (sess, m["size"])
            self.models.append((sess, m["size"], np.sqrt(m["weight"] / total), m["onnx"]))
        self.bg = ort.InferenceSession(f"{weights}/bg.onnx", opts, providers=self.providers)
        self.bank = dict(np.load(f"{weights}/cam_bank.npz"))
        # детектор ТС для автоопределения рамки в интерфейсе (необязателен)
        det = f"{weights}/detector.onnx"
        self.detector = ort.InferenceSession(det, opts, providers=self.providers) if os.path.exists(det) else None
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


# палитра тепловой карты: прозрачный -> фиолетовый -> малиновый -> жёлтый (цвета интерфейса)
_STOPS = np.array([[49, 15, 83], [138, 43, 226], [255, 0, 83], [255, 196, 0]], np.float32)


def _colorize(h, side):
    """h: 16x16 в [0, 1] -> RGBA PNG side x side (data URL)."""
    img = Image.fromarray((h * 255).astype(np.uint8), "L").resize((side, side), Image.BICUBIC)
    v = np.asarray(img, np.float32) / 255
    pos = v * (len(_STOPS) - 1)
    i = np.clip(pos.astype(int), 0, len(_STOPS) - 2)
    t = (pos - i)[..., None]
    rgb = _STOPS[i] * (1 - t) + _STOPS[i + 1] * t
    alpha = np.clip(v, 0, 1) ** 0.9 * 235
    rgba = np.concatenate([rgb, alpha[..., None]], -1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpeg(img, side):
    buf = io.BytesIO()
    img.convert("RGB").resize((side, side), Image.BICUBIC).save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def explain(engine, q_crop, c_crop, side=448):
    """Карта: какие участки запроса и кандидата дали сходство (вклад патча в косинус ViT-ветки)."""
    if engine.explainer is None:
        raise RuntimeError("нет weights/reid_explain.onnx")
    sess, size = engine.explainer
    x = np.stack([engine._vehicle(q_crop, size), engine._vehicle(c_crop, size)])
    emb, patches = sess.run(None, {"image": x})
    d = patches.shape[-1]
    g = int(round(patches.shape[1] ** 0.5))
    heat_q = patches[0] @ emb[1, d:]           # вклад участков запроса в сходство с кандидатом
    heat_c = patches[1] @ emb[0, d:]           # и наоборот
    out = {}
    for key, h, crop in (("query", heat_q, q_crop), ("candidate", heat_c, c_crop)):
        h = np.maximum(h, 0).reshape(g, g)
        # показываем только главные участки: нижние 60% вклада (ровный фон, асфальт) гасим,
        # верх шкалы — 99-й перцентиль, чтобы отдельные «шумные» токены DINOv2 не забивали шкалу
        lo, top = np.percentile(h, 60), np.percentile(h, 99)
        h = np.clip((h - lo) / max(top - lo, 1e-9), 0, 1) ** 1.3
        out[key] = {"image": _jpeg(crop, side), "heat": _colorize(h, side),
                    "aspect": round(crop.size[0] / max(1, crop.size[1]), 4)}   # w/h: показать без искажений
    out["similarity_vit"] = float(emb[0] @ emb[1])
    out["method"] = ("вклад каждого участка кадра в косинусную близость дообученной ViT-модели "
                     "(разложение сходства по патчам 14x14 px)")
    return out


VEHICLE = {3: "легковой", 4: "мотоцикл", 6: "автобус", 8: "грузовик"}   # классы COCO


def detect(engine, img, min_score=0.5, max_n=12):
    """Машины на кадре: [{x, y, w, h, score, kind}] в пикселях исходного кадра, крупные — первыми.
    Faster R-CNN MobileNetV3 (torchvision, COCO) на кадре 960x540 с полями."""
    if engine.detector is None:
        return []
    img = img.convert("RGB")
    W, H = img.size
    s = min(960 / W, 540 / H)
    small = img.resize((max(1, int(W * s)), max(1, int(H * s))), Image.BILINEAR)
    canvas = np.zeros((540, 960, 3), np.float32)
    canvas[:small.size[1], :small.size[0]] = np.asarray(small, np.float32) / 255
    try:
        boxes, labels, scores = engine.detector.run(None, {"image": canvas.transpose(2, 0, 1).copy()})
    except Exception:          # экспорт torchvision падает, если на кадре нет ни одного объекта
        return []
    out = []
    for b, lab, sc in zip(boxes / s, labels, scores):
        if int(lab) in VEHICLE and sc >= min_score:
            x0, y0, x1, y1 = [float(v) for v in b]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                out.append({"x": round(x0), "y": round(y0), "w": round(x1 - x0), "h": round(y1 - y0),
                            "score": round(float(sc), 3), "kind": VEHICLE[int(lab)]})
    # одна машина иногда находится дважды (разными классами) — оставляем более уверенную рамку
    out.sort(key=lambda r: -r["score"])
    keep = []
    for r in out:
        if all(_iou(r, k) < 0.55 for k in keep):
            keep.append(r)
    keep.sort(key=lambda r: -r["w"] * r["h"])
    return keep[:max_n]


def _iou(a, b):
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter or 1)
