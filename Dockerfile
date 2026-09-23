FROM python:3.11-slim

WORKDIR /app

# Install system dependencies ringan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Install PyTorch CPU-only secara khusus terlebih dahulu
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Copy dan install requirements yang sudah dirapikan
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]