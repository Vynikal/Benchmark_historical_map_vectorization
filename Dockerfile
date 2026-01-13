FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install numpy==1.24.4 \
	pandas==2.3.3 \
	opencv-python-headless==4.11.0.86 \
	scipy==1.15.3 \
	tqdm==4.67.1 \
	scikit-image==0.21.0 \
	Shapely==2.1.2 \
	scikit-learn==1.3.2 \
	setuptools==80.9.0 \
	psutil==5.8.0 \
	gdal==3.4.1
RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app