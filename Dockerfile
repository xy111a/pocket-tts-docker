FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir git+https://github.com/kyutai-labs/pocket-tts.git

COPY app.py .

EXPOSE 8002

CMD ["python", "app.py"]

