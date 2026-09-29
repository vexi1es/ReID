# Прототип на Render (Docker web service): API + веб-интерфейс в одном контейнере.
# Код и веса берутся из github.com/vexi1es/ReID (релиз v1.0); demo/ — готовая галерея и кадры-запросы,
# кладётся рядом с этим Dockerfile в приватном репозитории деплоя (данные организаторов не публикуются в коде).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ARG REF=main
RUN git clone --depth 1 --branch ${REF} https://github.com/vexi1es/ReID.git . && \
    pip install --no-cache-dir -r requirements-service.txt && \
    python get_weights.py
COPY demo/store/ /store/
COPY demo/ /demo/
ENV PYTHONPATH=/app:/app/service/api WEIGHTS=/app/weights STORE=/store WEB_DIR=/app/service/web \
    DEMO_DATA=/demo READONLY=1
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --app-dir service/api --host 0.0.0.0 --port ${PORT:-10000}"]
