FROM python:3.11-slim

WORKDIR /app

# Gerekli sistem kütüphaneleri (OpenCV için libgl1 vb.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gradio varsayılan portu 7860
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
