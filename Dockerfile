FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY app.py .
COPY config.py .
COPY utils.py .
COPY llm_service.py .
COPY semantic_search_service.py .
COPY ./data ./data
COPY ./templates ./templates


ENV GOOGLE_APPLICATION_CREDENTIALS /app/secrets/gcp_key.json


EXPOSE 5000


CMD ["python", "app.py"]