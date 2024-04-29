FROM python:3.10-slim

WORKDIR /app

COPY packages/requirements.txt .
RUN pip install -r requirements.txt

COPY . . 

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]