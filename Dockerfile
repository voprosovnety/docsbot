FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/

# The embedding model is downloaded on first run and cached here; mounting this
# as a volume keeps restarts fast.
RUN mkdir -p /app/models

CMD ["python", "-m", "bot.main"]
