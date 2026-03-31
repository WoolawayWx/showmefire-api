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
    cron \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Move venv to /opt/venv so it survives local volume mounts
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
# Updating PATH ensures specific venv binaries are used automatically
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version"
# Upgrade pip to avoid the notice in your logs
RUN pip install --upgrade pip

COPY requirements.lock.txt .
RUN python -m pip install --no-cache-dir -r requirements.lock.txt
COPY patches/rrfs.py /opt/venv/lib/python3.11/site-packages/herbie/models/rrfs.py
COPY . .


# Ensure a writable data directory (compose mounts ./api/data -> /app/data)
ENV DATA_DIR=/app/data
RUN mkdir -p ${DATA_DIR} && chown -R 1000:1000 ${DATA_DIR}
VOLUME ["/app/data"]

RUN echo "TZ=UTC" > /etc/cron.d/forecasts \
    && echo "30 14 * * * root /bin/bash /app/scripts/forecasts.sh >> /app/logs/cron.log 2>&1" >> /etc/cron.d/forecasts \
    && echo "" >> /etc/cron.d/forecasts \
    && chmod 0644 /etc/cron.d/forecasts

RUN echo "TZ=America/Chicago" > /etc/cron.d/validate \
    && echo "30 22 * * * root /bin/bash /app/scripts/validateForecast.sh >> /app/logs/valForecast.log 2>&1" >> /etc/cron.d/validate \
    && echo "" >> /etc/cron.d/validate \
    && chmod 0644 /etc/cron.d/validate

RUN echo "TZ=America/Chicago" > /etc/cron.d/createemptymaps \
    && echo "00 02 * * * root /bin/bash /app/scripts/create_empty_outlook_maps.sh >> /app/logs/create_empty_outlook_maps.log 2>&1" >> /etc/cron.d/createemptymaps \
    && echo "" >> /etc/cron.d/createemptymaps \
    && chmod 0644 /etc/cron.d/createemptymaps

# Copy entrypoint to run DB init before starting the server
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

EXPOSE 8000

# Use production-friendly CMD (no --reload)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

