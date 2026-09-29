"""
Тесты сервиса (API) на реальных весах. Запуск из корня решения:
    python -m pytest -q tests
Нужны: weights/ (python get_weights.py) и любой JPEG в tests/ (берётся из data/images, если есть).
"""
import glob
import importlib
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "service", "api")]


def _sample_jpeg():
    """Реальный кадр, если данные смонтированы, иначе синтетика (цветные полосы)."""
    files = glob.glob(os.path.join(ROOT, "data", "images", "*.jpg"))
    if files:
        return open(files[0], "rb").read(), (400, 200, 600, 400)
    img = Image.new("RGB", (1280, 720), (90, 90, 90))
    for i in range(0, 1280, 40):
        img.paste((i % 255, 60, 200 - i % 150), (i, 300, i + 20, 600))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue(), (300, 280, 500, 340)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.chdir(ROOT)
    os.environ["STORE"] = str(tmp_path_factory.mktemp("store"))
    os.environ.setdefault("WEIGHTS", os.path.join(ROOT, "weights"))
    if not os.path.exists(os.path.join(os.environ["WEIGHTS"], "reid.onnx")):
        pytest.skip("нет весов: python get_weights.py (или WEIGHTS=<папка с весами>)")
    os.environ.pop("READONLY", None)
    from fastapi.testclient import TestClient
    import app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app), app_module


def _form(bbox, **extra):
    x, y, w, h = bbox
    return {"x": str(x), "y": str(y), "w": str(w), "h": str(h), **{k: str(v) for k, v in extra.items()}}


def test_health(client):
    c, _ = client
    h = c.get("/api/health").json()
    assert h["status"] == "ok" and h["dim"] == 2304 and h["threshold"] == pytest.approx(0.30)


def test_embed_is_unit_vector(client):
    c, _ = client
    data, bbox = _sample_jpeg()
    r = c.post("/api/embed", files={"file": ("f.jpg", data, "image/jpeg")}, data=_form(bbox))
    assert r.status_code == 200
    e = np.array(r.json()["embedding"])
    assert e.shape == (2304,) and abs(np.linalg.norm(e) - 1) < 1e-3


def test_add_and_find_itself(client):
    c, _ = client
    data, bbox = _sample_jpeg()
    add = c.post("/api/gallery", files={"file": ("f.jpg", data, "image/jpeg")}, data=_form(bbox, label="self"))
    assert add.status_code == 200
    iid = add.json()["id"]
    r = c.post("/api/search", files={"file": ("f.jpg", data, "image/jpeg")},
               data=_form(bbox, exclude_same_camera="false", top_k=5)).json()
    assert r["ranking"][0]["id"] == iid and r["ranking"][0]["score"] > 0.99
    assert not r["refused"] and r["candidates"][0]["id"] == iid
    # тот же кадр с той же камеры при кросс-камерном поиске исключается -> галерея из 1 объекта даёт отказ
    r2 = c.post("/api/search", files={"file": ("f.jpg", data, "image/jpeg")}, data=_form(bbox)).json()
    assert r2["refused"] and all(m["id"] != iid for m in r2["ranking"])
    assert c.get(f"/api/gallery/{iid}/crop").status_code == 200


def test_threshold_controls_refusal(client):
    c, _ = client
    data, bbox = _sample_jpeg()
    r = c.post("/api/search", files={"file": ("f.jpg", data, "image/jpeg")},
               data=_form(bbox, exclude_same_camera="false", threshold=1.01)).json()
    assert r["refused"] and r["candidates"] == []


def test_bad_inputs(client):
    c, _ = client
    data, _ = _sample_jpeg()
    assert c.post("/api/search", files={"file": ("f.txt", b"hello", "text/plain")}).status_code == 415
    assert c.post("/api/search", files={"file": ("f.jpg", data, "image/jpeg")},
                  data=_form((0, 0, 3, 3))).status_code == 422
    assert c.post("/api/search", files={"file": ("f.jpg", data, "image/jpeg")},
                  data=_form((10, 10, 99999, 50))).status_code == 422
    assert c.get("/api/gallery/999999/crop").status_code == 404
    assert c.get("/api/demo/image/..%2F..%2Fetc").status_code in (400, 404)


def test_readonly_blocks_changes(client):
    c, app_module = client
    app_module.READONLY = True
    try:
        assert c.delete("/api/gallery").status_code == 403
        data, bbox = _sample_jpeg()
        assert c.post("/api/gallery", files={"file": ("f.jpg", data, "image/jpeg")},
                      data=_form(bbox)).status_code == 403
        assert c.get("/api/health").json()["readonly"] is True
    finally:
        app_module.READONLY = False


def test_delete_and_clear(client):
    c, _ = client
    data, bbox = _sample_jpeg()
    iid = c.post("/api/gallery", files={"file": ("f.jpg", data, "image/jpeg")}, data=_form(bbox)).json()["id"]
    assert c.delete(f"/api/gallery/{iid}").status_code == 200
    assert c.delete("/api/gallery").json()["gallery"] == 0
    assert c.get("/api/health").json()["gallery"] == 0


def test_explain_heatmap(client):
    c, app_module = client
    if app_module.engine.explainer is None:
        pytest.skip("нет weights/reid_explain.onnx")
    data, bbox = _sample_jpeg()
    iid = c.post("/api/gallery", files={"file": ("f.jpg", data, "image/jpeg")}, data=_form(bbox)).json()["id"]
    r = c.post("/api/explain", files={"file": ("f.jpg", data, "image/jpeg")}, data=_form(bbox, gallery_id=iid))
    assert r.status_code == 200
    j = r.json()
    assert j["query"]["heat"].startswith("data:image/png") and j["candidate"]["image"].startswith("data:image/jpeg")
    assert j["similarity_vit"] > 0.99
    assert c.post("/api/explain", files={"file": ("f.jpg", data, "image/jpeg")},
                  data=_form(bbox, gallery_id=987654)).status_code == 404
