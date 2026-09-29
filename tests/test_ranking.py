"""
Тесты логики ранжирования и артефактов сдачи (без нейросетей, быстрые):
    python -m pytest -q tests
"""
import csv
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SUB = os.path.join(ROOT, "submission") if os.path.isdir(os.path.join(ROOT, "submission")) else None
DATA = os.environ.get("REID_DATA", os.path.join(ROOT, "data"))

from reid.rerank import k_reciprocal  # noqa: E402
from reid.submit import candidates_df, write_submission  # noqa: E402


def test_k_reciprocal_prefers_mutual_neighbours():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 16))
    q = base + 0.05 * rng.normal(size=base.shape)
    g = np.concatenate([base + 0.05 * rng.normal(size=base.shape), rng.normal(size=(20, 16))])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    d = k_reciprocal(q, g, k1=4, k2=2, lam=0.3)
    assert d.shape == (5, 25)
    assert (d.argmin(1) == np.arange(5)).all()


def test_candidates_threshold_and_order(tmp_path):
    q_ids, g_ids = ["q1", "q2"], ["g1", "g2", "g3"]
    sims = np.array([[0.9, 0.1, 0.2], [0.1, 0.25, 0.2]])
    order = np.array([[0, 2, 1], [1, 2, 0]])
    c = candidates_df(q_ids, g_ids, sims, order, 0.30)
    assert c.to_dict("records") == [{"query_id": "q1", "gallery_id": "g1", "confidence": 0.9}]
    write_submission(tmp_path / "s.csv", q_ids, g_ids, order)
    rows = list(csv.reader(open(tmp_path / "s.csv", encoding="utf-8")))
    assert rows == [["q1", "g1", "g3", "g2"], ["q2", "g2", "g3", "g1"]]


@pytest.mark.skipif(SUB is None or not os.path.exists(os.path.join(DATA, "test_query.csv")),
                    reason="нужны submission/ и data/test_*.csv")
def test_submission_artifacts():
    q = pd.read_csv(os.path.join(DATA, "test_query.csv"))
    g = pd.read_csv(os.path.join(DATA, "test_gallery.csv"))
    rows = list(csv.reader(open(os.path.join(SUB, "submission.csv"), encoding="utf-8")))
    gal = set(g.image_id)
    assert [r[0] for r in rows] == list(q.image_id)
    assert all(len(r) == 11 and len(set(r[1:])) == 10 and set(r[1:]) <= gal for r in rows)
    emb = np.load(os.path.join(SUB, "embeddings.npy"))
    assert emb.shape == (len(q) + len(g), 2304) and emb.dtype == np.float32
    assert np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-4)
    c = pd.read_csv(os.path.join(SUB, "candidates.csv"))
    assert list(c.columns) == ["query_id", "gallery_id", "confidence"]
    assert c.query_id.is_unique and (c.confidence >= 0.30).all() and set(c.gallery_id) <= gal
    # кандидат = первый в выдаче submission.csv
    first = {r[0]: r[1] for r in rows}
    assert all(first[a] == b for a, b in zip(c.query_id, c.gallery_id))


@pytest.mark.skipif(SUB is None or not os.path.exists(os.path.join(SUB, "ranking_inputs.csv")),
                    reason="нужен submission/ranking_inputs.csv")
def test_ranking_reproducible_from_embeddings(tmp_path):
    out = subprocess.run([sys.executable, "reproduce_ranking.py", "--sub", SUB, "--data", DATA,
                          "--out", str(tmp_path), "--check"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("совпадает байт в байт") == 2, out.stdout
