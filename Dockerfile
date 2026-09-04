FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY backend.py .
COPY static ./static

CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000}"]