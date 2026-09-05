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
    gh \
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
COPY requirements.pyretechnics.txt .
# Pyretechnics pins NumPy 1.24.x, while goes2go requires NumPy 2.2.5+.
# Its surface-fire extension is compatible with the locked NumPy 2.2.6,
# so install the pinned package without asking pip to resolve that stale pin.
RUN python -m pip install --no-cache-dir --no-deps -r requirements.pyretechnics.txt \
    && python -c "import numpy, pyretechnics.surface_fire; assert numpy.__version__ == '2.2.6'"
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

# Begin checking the 09z HRRR f05-f18 window just after 12z. The job itself
# waits for every required frame and field before forecast generation.
RUN echo "TZ=UTC" > /etc/cron.d/forecasts09z \
    && echo "05 12 * * * root /bin/bash /app/scripts/forecasts_09z.sh >> /app/logs/cron09z.log 2>&1" >> /etc/cron.d/forecasts09z \
    && echo "" >> /etc/cron.d/forecasts09z \
    && chmod 0644 /etc/cron.d/forecasts09z

# Verification covers 10:00-21:00 Central. Run at 04:30 UTC on the following
# UTC day: 22:30 CST / 23:30 CDT, safely after the local observation window.
RUN echo "TZ=UTC" > /etc/cron.d/validate \
    && echo "30 04 * * * root /bin/bash /app/scripts/validateForecast.sh >> /app/logs/valForecast.log 2>&1" >> /etc/cron.d/validate \
    && echo "" >> /etc/cron.d/validate \
    && chmod 0644 /etc/cron.d/validate

RUN echo "TZ=America/Chicago" > /etc/cron.d/createemptymaps \
    && echo "00 02 * * * root /bin/bash /app/scripts/create_empty_outlook_maps.sh >> /app/logs/create_empty_outlook_maps.log 2>&1" >> /etc/cron.d/createemptymaps \
    && echo "" >> /etc/cron.d/createemptymaps \
    && chmod 0644 /etc/cron.d/createemptymaps

RUN echo "TZ=UTC" > /etc/cron.d/maps \
    && echo "*/15 * * * * root /bin/bash /app/scripts/maps.sh >> /app/logs/maps.log 2>&1" >> /etc/cron.d/maps \
    && echo "" >> /etc/cron.d/maps \
    && chmod 0644 /etc/cron.d/maps

# Copy entrypoint to run DB init before starting the server
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

EXPOSE 8000

# Use production-friendly CMD (no --reload)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
