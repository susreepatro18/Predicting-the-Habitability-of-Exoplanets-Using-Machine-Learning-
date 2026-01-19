FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
