FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code if needed, but for now we'll just copy the service code
COPY . .

CMD ["python", "main.py"]
