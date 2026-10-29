# Lightweight official Python image (the code requires Python 3.10+)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000

WORKDIR /app

# Copy requirements first so Docker can cache these layers
COPY requirements.txt .

# 1. CPU-only PyTorch, installed before the other packages so pip keeps this build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Remaining pinned dependencies
RUN pip install -r requirements.txt

# 3. Application code (.dockerignore keeps secrets and local data out)
COPY . .

RUN mkdir -p /app/vector_store

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]