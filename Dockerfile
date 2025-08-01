FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "document_intelligence.py"]
