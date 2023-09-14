ARG METRIC_IMAGE=None
FROM ${METRIC_IMAGE} as base


RUN apt-get update -q 
RUN apt-get install build-essential -qqy
RUN pip3 install opencv-python
RUN pip3 install numpy
RUN pip3 install tqdm
RUN pip3 install av
RUN pip3 install scikit-image

COPY run.py /run.py
