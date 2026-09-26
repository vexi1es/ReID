# Прототип на Hugging Face Spaces (Docker SDK): API + веб-клиент в одном контейнере, порт 7860.
# Сборка берёт код и веса из GitHub (github.com/vexi1es/ReID, релиз v1.0).
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
RUN git clone --depth 1 https://github.com/vexi1es/ReID.git . && \
    pip install --no-cache-dir -r requirements-service.txt && \
    python get_weights.py
# demo/: готовая галерея (store/: индекс + превью ТС) и кадры-запросы — лежат только в репозитории Space
COPY --chown=user demo/store/ /home/user/store/
COPY --chown=user demo/ /home/user/demo/
RUN chown -R user /home/user/app
USER user
ENV PYTHONPATH=/home/user/app:/home/user/app/service/api WEIGHTS=/home/user/app/weights \
    STORE=/home/user/store WEB_DIR=/home/user/app/service/web DEMO_DATA=/home/user/demo READONLY=1
EXPOSE 7860
CMD ["uvicorn", "app:app", "--app-dir", "service/api", "--host", "0.0.0.0", "--port", "7860"]
