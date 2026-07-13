FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 10000

CMD ["gunicorn", "--workers=1", "--threads=2", "--timeout=300", "--preload", "--bind", "0.0.0.0:10000", "app:app"]
