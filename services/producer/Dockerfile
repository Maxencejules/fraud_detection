FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/producer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/producer /app/services/producer
COPY shared /app/shared
COPY data /app/data

WORKDIR /app/services/producer
CMD ["python", "main.py"]
