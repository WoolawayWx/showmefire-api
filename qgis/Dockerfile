FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      qgis-server spawn-fcgi xvfb fonts-dejavu-core ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV QGIS_PREFIX_PATH=/usr \
    QGIS_PLUGINPATH=/io/plugins \
    QGIS_SERVER_LOG_STDERR=1 \
    QGIS_SERVER_LOG_LEVEL=2 \
    QGIS_SERVER_PARALLEL_RENDERING=true \
    QGIS_SERVER_MAX_THREADS=2 \
    QGIS_AUTH_DB_DIR_PATH=/tmp/ \
    HOME=/var/lib/qgis \
    DISPLAY=:99 \
    LANG=en_US.UTF-8

WORKDIR /var/lib/qgis
EXPOSE 9993
