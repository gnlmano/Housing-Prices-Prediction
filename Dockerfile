FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# COPY . .

COPY src /app/src
COPY api /app/api
COPY models /app/models
COPY requirements.txt /app/requirements.txt


EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
