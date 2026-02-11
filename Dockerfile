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
    curl \
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

COPY patches/rrfs.py /opt/venv/lib/python3.11/site-packages/herbie/models/rrfs.py
# Ensure a writable data directory (compose mounts ./api/data -> /app/data)
ENV DATA_DIR=/app/data
RUN mkdir -p ${DATA_DIR} && chown -R 1000:1000 ${DATA_DIR}
VOLUME ["/app/data"]

# Copy entrypoint to run DB init before starting the server
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

EXPOSE 8000

# Use production-friendly CMD (no --reload)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]