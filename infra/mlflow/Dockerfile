# The official MLflow image ships without a PostgreSQL driver, so a
# `postgresql://` backend store fails at start-up. Add the driver and run
# the tracking server as an unprivileged user.
FROM ghcr.io/mlflow/mlflow:v3.16.1

RUN pip install --no-cache-dir psycopg2-binary==2.9.13 \
    && useradd --system --uid 10001 --home-dir /mlflow --shell /usr/sbin/nologin mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown -R mlflow /mlflow

USER mlflow
WORKDIR /mlflow
EXPOSE 5000
