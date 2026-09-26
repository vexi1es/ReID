"""
Хранилище галереи: метаданные и эмбеддинги — в SQLite (файл на томе), поиск — FAISS.

INDEX=flat (по умолчанию) — точный поиск по косинусу; INDEX=hnsw — приближённый (HNSW),
для галерей порядка 10^6 (см. bench_ann.py). Индекс пересобирается из SQLite при старте,
поэтому единственный источник истины — база.
"""
import os
import sqlite3
import threading
import time

import faiss
import numpy as np


class Store:
    def __init__(self, root, dim, kind=None):
        os.makedirs(f"{root}/crops", exist_ok=True)
        self.root, self.dim = root, dim
        self.kind = kind or os.environ.get("INDEX", "flat")
        self.lock = threading.Lock()
        self.db = sqlite3.connect(f"{root}/gallery.db", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, camera INTEGER, source TEXT,
            x REAL, y REAL, w REAL, h REAL, created REAL, emb BLOB)""")
        self.db.commit()
        self._rebuild()

    def _new_index(self):
        if self.kind == "hnsw":
            base = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efSearch = 128
        else:
            base = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIDMap(base)

    def _rebuild(self):
        self.index = self._new_index()
        rows = self.db.execute("SELECT id, emb FROM items").fetchall()
        if rows:
            ids = np.array([r[0] for r in rows], np.int64)
            embs = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
            self.index.add_with_ids(embs, ids)

    def add(self, emb, crop, label=None, camera=None, source="upload", bbox=(0, 0, 0, 0)):
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO items (label, camera, source, x, y, w, h, created, emb) VALUES (?,?,?,?,?,?,?,?,?)",
                (label, camera, source, *map(float, bbox), time.time(), emb.astype(np.float32).tobytes()))
            self.db.commit()
            iid = cur.lastrowid
            self.index.add_with_ids(emb[None].astype(np.float32), np.array([iid], np.int64))
        crop.convert("RGB").resize(_thumb(crop.size)).save(f"{self.root}/crops/{iid}.jpg", quality=90)
        return iid

    def delete(self, iid):
        with self.lock:
            self.db.execute("DELETE FROM items WHERE id=?", (iid,))
            self.db.commit()
            self._rebuild()          # HNSW не умеет удалять — проще пересобрать
        try:
            os.remove(f"{self.root}/crops/{iid}.jpg")
        except OSError:
            pass

    def clear(self):
        with self.lock:
            self.db.execute("DELETE FROM items")
            self.db.commit()
            self._rebuild()
        for f in os.listdir(f"{self.root}/crops"):
            os.remove(f"{self.root}/crops/{f}")

    def search(self, emb, k):
        with self.lock:
            n = self.index.ntotal
            if n == 0:
                return []
            D, I = self.index.search(emb[None].astype(np.float32), min(k, n))
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]

    def meta(self, ids):
        if not ids:
            return {}
        q = f"SELECT id, label, camera, source FROM items WHERE id IN ({','.join('?' * len(ids))})"
        return {r[0]: dict(id=r[0], label=r[1], camera=r[2], source=r[3]) for r in self.db.execute(q, ids)}

    def list(self, offset=0, limit=50):
        rows = self.db.execute("SELECT id, label, camera, source FROM items ORDER BY id LIMIT ? OFFSET ?",
                               (limit, offset)).fetchall()
        return [dict(id=r[0], label=r[1], camera=r[2], source=r[3]) for r in rows]

    def count(self):
        return self.index.ntotal


def _thumb(size, side=256):
    w, h = size
    s = side / max(w, h)
    return max(1, int(w * s)), max(1, int(h * s))
