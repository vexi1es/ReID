"""
REST API сервиса «цифровой признак ТС». Спецификация OpenAPI — /api/openapi.json, Swagger UI — /api/docs.

Этапы (раздел 4 ТЗ): получение (изображение + bbox, валидация) -> обработка (эмбеддинг ТС) ->
анализ (поиск по FAISS, кадры той же камеры исключаются) -> результат (top-N или отказ).

    uvicorn app:app --host 0.0.0.0 --port 8000        (из service/api, PYTHONPATH = корень решения)
"""
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from engine import Engine, crop_box
from store import Store

WEIGHTS = os.environ.get("WEIGHTS", "weights")
STORE = os.environ.get("STORE", "store")
DEMO_DATA = os.environ.get("DEMO_DATA", "/data")          # test_gallery.csv + images/ для демо-галереи
MAX_BYTES = 25 * 2 ** 20
READONLY = os.environ.get("READONLY") == "1"      # публичное демо: галерею менять нельзя

app = FastAPI(title="Цифровой признак ТС", version="1.0",
              description="Поиск того же транспортного средства на кадрах других камер без госномера. "
                          "Эмбеддинг: ансамбль DINOv2 ViT-B/14 + DINOv3 ConvNeXt-S, дообученных на train.",
              docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
engine = Engine(WEIGHTS)
store = Store(STORE, dim=int(engine.embed(Image.new("RGB", (64, 64)), (0, 0, 64, 64), flip=False)[0].size))
demo_state = {"running": False, "done": 0, "total": 0, "error": None}


class Match(BaseModel):
    id: int
    score: float = None
    label: Optional[str] = None
    camera: Optional[int] = None
    accepted: bool = False
    crop_url: str = ""


class SearchResult(BaseModel):
    refused: bool
    threshold: float
    query_camera: Optional[int]
    candidates: List[Match]
    ranking: List[Match]
    ms: float


class Item(BaseModel):
    id: int
    label: Optional[str] = None
    camera: Optional[int] = None
    source: Optional[str] = None


def read_image(file: UploadFile):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "файл больше 25 МБ")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "не удалось прочитать изображение (нужен JPEG или PNG)")
    return img


def full_bbox(img, x, y, w, h):
    if w is None or h is None:          # bbox не задан — весь кадр (уже вырезанное ТС)
        return 0.0, 0.0, float(img.size[0]), float(img.size[1])
    return float(x or 0), float(y or 0), float(w), float(h)


def writable():
    if READONLY:
        raise HTTPException(403, "демо-режим: галерея только для чтения")


def embed_or_400(img, bbox):
    try:
        return engine.embed(img, bbox)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/health", summary="Состояние сервиса")
def health():
    return {"status": "ok", "gallery": store.count(), "readonly": READONLY, "index": store.kind, "providers": engine.providers,
            "models": [m[3] for m in engine.models], "threshold": engine.threshold, "dim": store.dim}


@app.post("/api/search", response_model=SearchResult, summary="Найти ТС в галерее (top-N или отказ)")
def search(file: UploadFile = File(..., description="кадр JPEG/PNG"),
           x: Optional[float] = Form(None), y: Optional[float] = Form(None),
           w: Optional[float] = Form(None), h: Optional[float] = Form(None),
           top_k: int = Form(10, ge=1, le=100),
           camera_id: Optional[int] = Form(None, description="камера запроса; если не задана — определяется по фону"),
           exclude_same_camera: bool = Form(True, description="исключать кадры той же камеры (кросс-камерный поиск)"),
           threshold: Optional[float] = Form(None, description="порог отказа; по умолчанию обоснованный 0.30")):
    t0 = time.time()
    img = read_image(file)
    bbox = full_bbox(img, x, y, w, h)
    emb, _ = embed_or_400(img, bbox)
    thr = engine.threshold if threshold is None else threshold
    cam = camera_id if camera_id is not None else (engine.camera(img, bbox) if w is not None else None)
    hits = store.search(emb, top_k + 200)
    meta = store.meta([i for i, _ in hits])
    ranking = []
    for i, s in hits:
        m = meta.get(i, {})
        if exclude_same_camera and cam is not None and m.get("camera") == cam:
            continue
        ranking.append(Match(id=i, score=round(s, 4), label=m.get("label"), camera=m.get("camera"),
                             accepted=s >= thr, crop_url=f"/api/gallery/{i}/crop"))
        if len(ranking) == top_k:
            break
    # режим кандидатов: top-1 с уверенностью не ниже порога, иначе отказ (как candidates.csv)
    cands = [r for r in ranking[:1] if r.accepted]
    return SearchResult(refused=not cands, threshold=thr, query_camera=cam, candidates=cands,
                        ranking=ranking, ms=round((time.time() - t0) * 1000, 1))


@app.post("/api/embed", summary="Эмбеддинг ТС (цифровой признак)")
def embed(file: UploadFile = File(...), x: Optional[float] = Form(None), y: Optional[float] = Form(None),
          w: Optional[float] = Form(None), h: Optional[float] = Form(None)):
    img = read_image(file)
    emb, _ = embed_or_400(img, full_bbox(img, x, y, w, h))
    return {"dim": int(emb.size), "embedding": [round(float(v), 6) for v in emb]}


@app.post("/api/gallery", response_model=Item, summary="Добавить ТС в галерею")
def add(file: UploadFile = File(...), x: Optional[float] = Form(None), y: Optional[float] = Form(None),
        w: Optional[float] = Form(None), h: Optional[float] = Form(None),
        label: Optional[str] = Form(None), camera_id: Optional[int] = Form(None)):
    writable()
    img = read_image(file)
    bbox = full_bbox(img, x, y, w, h)
    emb, crop = embed_or_400(img, bbox)
    cam = camera_id if camera_id is not None else (engine.camera(img, bbox) if w is not None else None)
    iid = store.add(emb, crop, label=label, camera=cam, source="upload", bbox=bbox)
    return Item(id=iid, label=label, camera=cam, source="upload")


@app.get("/api/gallery", response_model=List[Item], summary="Список галереи")
def list_gallery(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
    return store.list(offset, limit)


@app.get("/api/gallery/{iid}/crop", summary="Превью ТС из галереи", response_class=FileResponse)
def crop(iid: int):
    p = f"{STORE}/crops/{iid}.jpg"
    if not os.path.exists(p):
        raise HTTPException(404, "нет такого объекта")
    return FileResponse(p, media_type="image/jpeg")


@app.delete("/api/gallery/{iid}", summary="Удалить объект из галереи")
def delete(iid: int):
    writable()
    store.delete(iid)
    return {"deleted": iid}


@app.delete("/api/gallery", summary="Очистить галерею")
def clear():
    writable()
    store.clear()
    return {"gallery": 0}


def _load_demo(limit):
    try:
        g = pd.read_csv(f"{DEMO_DATA}/test_gallery.csv")
        if limit:
            g = g.head(limit)
        demo_state.update(running=True, done=0, total=len(g), error=None)
        rows = list(g.itertuples())

        def prep(r):                     # чтение кадра и подготовка — параллельно, модели — батчем
            img = Image.open(f"{DEMO_DATA}/images/{r.image_id}.jpg").convert("RGB")
            bbox = (r.x, r.y, r.w, r.h)
            return crop_box(img, bbox), engine.background(img, bbox), bbox

        with ThreadPoolExecutor(4) as pool:
            for i in range(0, len(rows), 32):
                chunk = rows[i:i + 32]
                prepared = list(pool.map(prep, chunk))
                embs = engine.embed_crops([p[0] for p in prepared])
                cams = engine.cameras([p[1] for p in prepared])
                for r, (crop, _, bbox), emb, cam in zip(chunk, prepared, embs, cams):
                    store.add(emb, crop, label=r.image_id, camera=cam, source="test_gallery", bbox=bbox)
                demo_state["done"] += len(chunk)
    except Exception as e:  # noqa: BLE001 — показать причину в интерфейсе
        demo_state["error"] = str(e)
    finally:
        demo_state["running"] = False


@app.post("/api/demo/load", summary="Загрузить test_gallery.csv из смонтированных данных в галерею")
def load_demo(limit: int = Query(0, ge=0, description="0 — всю галерею")):
    writable()
    if demo_state["running"]:
        raise HTTPException(409, "загрузка уже идёт")
    if not os.path.exists(f"{DEMO_DATA}/test_gallery.csv"):
        raise HTTPException(404, f"нет {DEMO_DATA}/test_gallery.csv — смонтируйте данные в /data")
    threading.Thread(target=_load_demo, args=(limit,), daemon=True).start()
    return {"started": True}


@app.get("/api/demo/status", summary="Прогресс загрузки демо-галереи")
def demo_status():
    return demo_state


@app.get("/api/demo/queries", summary="Примеры запросов из test_query.csv (кадр + bbox)")
def demo_queries(limit: int = Query(24, ge=1, le=200), seed: int = Query(0)):
    p = f"{DEMO_DATA}/test_query.csv"
    if not os.path.exists(p):
        return []
    q = pd.read_csv(p)
    q = q.sample(min(limit, len(q)), random_state=seed)
    return [dict(image_id=r.image_id, x=r.x, y=r.y, w=r.w, h=r.h, url=f"/api/demo/image/{r.image_id}")
            for r in q.itertuples()]


@app.get("/api/demo/image/{image_id}", summary="Полный кадр из демо-данных", response_class=FileResponse)
def demo_image(image_id: str):
    if not image_id.isalnum():
        raise HTTPException(400, "некорректный image_id")
    p = f"{DEMO_DATA}/images/{image_id}.jpg"
    if not os.path.exists(p):
        raise HTTPException(404, "нет кадра")
    return FileResponse(p, media_type="image/jpeg")


if os.environ.get("WEB_DIR"):       # локальный запуск без nginx: API сам отдаёт веб-интерфейс
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=os.environ["WEB_DIR"], html=True), name="web")
