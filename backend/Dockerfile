# Pull base image
FROM python:3.11.4-slim-bullseye

# Set environment variables
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV COLUMNS 80

# install Debian and other dependencies that are required to run python apps(eg. git, python-magic).
RUN apt-get clean && apt-get update  --fix-missing \
  && apt-get install -y \
    chrpath \
    ffmpeg \
    gcc \
    gettext \
    git \
    libfontconfig1 \
    libfontconfig1-dev \
    libfreetype6 \
    libfreetype6-dev \
    libmagic-dev \
    libpq-dev \
    libssl-dev \
    libxft-dev \
    nano \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory for Docker image
WORKDIR /code/

# Install dependencies
COPY . .
RUN pip install -r requirements.txt