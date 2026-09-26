# Backend инференса: FastAPI + onnxruntime + FAISS. Контекст сборки — корень решения.
FROM python:3.12-slim
ARG ORT=onnxruntime==1.23.2
WORKDIR /app
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt && \
    if [ "$ORT" != "onnxruntime==1.23.2" ]; then pip uninstall -y onnxruntime && pip install --no-cache-dir "$ORT"; fi
COPY infer.py .
COPY reid/__init__.py reid/data.py reid/rerank.py reid/submit.py reid/
COPY service/api/ service/api/
COPY weights/ weights/
ENV PYTHONPATH=/app:/app/service/api WEIGHTS=/app/weights STORE=/store DEMO_DATA=/data
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "app:app", "--app-dir", "service/api", "--host", "0.0.0.0", "--port", "8000"]
