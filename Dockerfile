FROM python:3.12-slim

WORKDIR /app
COPY requirements-infer.txt .
# ORT=onnxruntime-gpu[cuda,cudnn]==1.23.2 — сборка для GPU (docker-compose.gpu.yml)
ARG ORT=onnxruntime==1.23.2
RUN pip install --no-cache-dir -r requirements-infer.txt && \
    if [ "$ORT" != "onnxruntime==1.23.2" ]; then pip uninstall -y onnxruntime && pip install --no-cache-dir "$ORT"; fi

COPY infer.py .
COPY reid/__init__.py reid/data.py reid/rerank.py reid/submit.py reid/
COPY weights/reid.onnx weights/reid_cnx.onnx weights/bg.onnx weights/view.onnx weights/cam_bank.npz weights/meta.json weights/

ENTRYPOINT ["python", "infer.py", "--data", "/data", "--out", "/out"]
