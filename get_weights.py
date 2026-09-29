"""
Скачивает веса моделей (632 МБ) из GitHub Releases в ./weights. Нужен только один раз перед сборкой.
После этого сборка и инференс работают без интернета.

    python get_weights.py
"""
import hashlib
import os
import sys
import urllib.request

REPO = "vexi1es/ReID"
TAG = "v1.0"
FILES = ["reid.onnx", "reid_cnx.onnx", "bg.onnx", "view.onnx", "cam_bank.npz",
         "reid_explain.onnx", "detector.onnx"]   # два последних — только для сервиса (карта внимания, автоопределение)


def main():
    os.makedirs("weights", exist_ok=True)
    for name in FILES:
        dst = f"weights/{name}"
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"{dst}: уже есть")
            continue
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{name}"
        print(f"{url} -> {dst}", flush=True)
        tmp = dst + ".part"
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total, done = int(r.headers.get("Content-Length", 0)), 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    sys.stdout.write(f"\r  {done / 2**20:.0f} / {total / 2**20:.0f} МБ")
                    sys.stdout.flush()
        os.replace(tmp, dst)
        print()
    sums = open("weights/SHA256SUMS").read().split("\n") if os.path.exists("weights/SHA256SUMS") else []
    for line in filter(None, sums):
        h, name = line.split()
        name = name.lstrip("*")
        ok = hashlib.sha256(open(f"weights/{name}", "rb").read()).hexdigest() == h
        print(f"{name}: {'OK' if ok else 'КОНТРОЛЬНАЯ СУММА НЕ СОВПАЛА'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
