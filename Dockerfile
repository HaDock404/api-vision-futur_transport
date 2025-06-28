FROM python:3.10-slim

WORKDIR /app

COPY packages/requirements.txt .

# INSTALL DEPENDENCIES REQUIRED FOR OPENCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY . .

EXPOSE 8080

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]