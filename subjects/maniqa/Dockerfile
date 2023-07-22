FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget https://github.com/IIGROUP/MANIQA/releases/download/Koniq10k/ckpt_koniq10k.pt

COPY src /src

COPY patches/maniqa.patch /src

RUN cd /src && dos2unix maniqa.patch && dos2unix models/maniqa.py && patch -Np1 < maniqa.patch

RUN pip install einops

COPY test.py /

COPY model.py /src
COPY config.json /src
