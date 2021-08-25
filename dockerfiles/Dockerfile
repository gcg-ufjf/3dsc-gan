FROM nvcr.io/nvidia/tensorflow:18.10-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        apt-utils \
        nfs-common \
        cifs-utils \
        ssh \
        libsm6 \ 
        libxext6 \ 
        libxrender-dev

RUN pip install --upgrade pip

# Removing default numpy
RUN pip uninstall -y numpy

WORKDIR /tmp/
COPY requirements.txt /tmp/
RUN pip install -r requirements.txt
RUN rm requirements.txt

WORKDIR /workspace

CMD ["/bin/bash"]
