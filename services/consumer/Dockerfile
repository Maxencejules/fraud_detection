FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY services/consumer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY services/consumer /app/services/consumer
COPY shared /app/shared
COPY data /app/data

WORKDIR /app/services/consumer
CMD ["python", "main.py"]
