FROM python:3.12-slim

# ffmpeg encodes MP3/M4B; espeak-ng is a zero-download TTS engine and Kokoro's phonemizer fallback
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg espeak-ng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt requirements-tts.txt ./
ARG WITH_KOKORO=0
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_KOKORO" = "1" ]; then pip install --no-cache-dir -r requirements-tts.txt; fi

COPY app ./app
COPY web ./web
COPY assets ./assets
COPY main.py ./

RUN useradd --create-home sonic && mkdir -p /data && chown sonic /data
USER sonic
ENV SONICTALES_DATA=/data
VOLUME /data
EXPOSE 8000
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
