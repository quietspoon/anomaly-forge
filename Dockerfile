FROM python:3.11-slim

# Install system dependencies needed to compile numpy, scipy, dtaidistance
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install Python dependencies (separate layer for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default: run pytest
CMD ["pytest", "tests/", "-v", "--tb=short"]
