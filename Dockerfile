# Dockerfile for Hugging Face Spaces (Docker SDK)
# Using a pinned Debian base avoids the shifting Python/package-naming issues
# we hit on Streamlit Community Cloud (which uses a bleeding-edge Python 3.14
# + Debian trixie combo that keeps renaming/breaking system packages).

FROM python:3.11-slim

# System libraries needed by OpenCV (non-headless, pulled in by mediapipe)
# and MediaPipe's own native/GL dependencies. Same libraries we identified
# needed on Streamlit Cloud, using this base image's stable package names.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces expects the app to listen on port 7860
EXPOSE 7860

# --server.address=0.0.0.0 is required so the container accepts external
# connections (default is localhost-only, which HF Spaces can't reach).
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
