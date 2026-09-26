"""Локальный запуск сервиса без Docker (API + веб на одном порту): python service/dev.py [порт]"""
import glob
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
sys.path[:0] = [root, os.path.join(root, "service", "api")]
os.environ.setdefault("STORE", os.path.join(root, "..", "scratch_store"))
os.environ.setdefault("WEB_DIR", os.path.join(root, "service", "web"))
data = glob.glob(os.path.join(root, "..", "*", "dataset", "test_gallery.csv"))
if data:
    os.environ.setdefault("DEMO_DATA", os.path.dirname(data[0]))

import uvicorn  # noqa: E402

uvicorn.run("app:app", host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 18080)
