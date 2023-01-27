FROM python:3.9-slim-buster

ARG NOMBRE_PROYECTO

ENV EP="$NOMBRE_PROYECTO/src/ds/api.py"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -r requirements.txt --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential

RUN mkdir $NOMBRE_PROYECTO
RUN mkdir $NOMBRE_PROYECTO/data
RUN mkdir $NOMBRE_PROYECTO/data/input
RUN mkdir $NOMBRE_PROYECTO/data/output
RUN mkdir $NOMBRE_PROYECTO/data/processed

# RUN python3 -m pip install $NOMBRE_PROYECTO/GDAL-3.4.1-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.whl

# RUN export PYTHONPATH="$PYTHONPATH:/src"

# ENTRYPOINT "python3" $EP
