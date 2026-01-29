FROM python:3.11-slim

# Install system dependencies for GIS and PostgreSQL
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgdal-dev \
    libproj-dev \
    gdal-bin \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Move venv to /opt/venv so it survives local volume mounts
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
# Updating PATH ensures specific venv binaries are used automatically
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip to avoid the notice in your logs
RUN pip install --upgrade pip

COPY requirements.txt .

# Install dependencies (now tracking into /app/venv)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]