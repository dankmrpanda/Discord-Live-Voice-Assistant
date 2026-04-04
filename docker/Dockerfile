FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime deps for audio + build deps for packages with C extensions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# THE FIX: Download models as ROOT so they can be written to the site-packages directory
RUN python -c "import openwakeword; openwakeword.utils.download_models()"

RUN useradd --create-home --shell /usr/sbin/nologin botuser \
    && chown -R botuser:botuser /app

# Create non-root user for security and take ownership of /app
RUN useradd --create-home --shell /bin/bash botuser && \
    chown -R botuser:botuser /app

# Switch to non-root user for execution
USER botuser

# Run the bot
CMD ["python", "-m", "src.main"]
