FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY flask_backend/requirements.txt /app/flask_backend/requirements.txt
RUN pip install --no-cache-dir -r /app/flask_backend/requirements.txt

COPY flask_backend /app/flask_backend

EXPOSE 8000

CMD ["sh", "-c", "uvicorn flask_backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]