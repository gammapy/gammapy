# This is the Dockerfile to run Gammapy on Binder
#

FROM continuumio/miniconda3
MAINTAINER Gammapy developers <gammapy@googlegroups.com>

# compilers
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip

# install dependencies - including the stable version of Gammapy
COPY binder.py tmp/
RUN curl -o tmp/environment.yml https://gammapy.org/download/install/gammapy-0.8-environment.yml

WORKDIR tmp/
RUN conda update conda
RUN conda install -q -y pyyaml
RUN python binder.py

# add gammapy user running the jupyter notebook process
ENV NB_USER gammapy
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# setting ownerships
USER root
RUN chown -R ${NB_UID} ${HOME}

# download tutorials and datasets
RUN gammapy download tutorials --out=${HOME}/gammapy-tutorials --release=0.8
RUN git clone https://github.com/gammapy/gammapy-cat.git ${HOME}/gammapy-cat
RUN git clone https://github.com/gammapy/gammapy-fermi-lat-data.git ${HOME}/gammapy-fermi-lat-data

# start JupyterLab server in tutorials dir
USER ${NB_USER}
WORKDIR ${HOME}/gammapy-tutorials/notebooks-0.8

# env vars used in tutorials
ENV GAMMAPY_DATA ${HOME}/gammapy-tutorials/datasets
ENV GAMMAPY_CAT ${HOME}/gammapy-cat
ENV GAMMAPY_FERMI_LAT_DATA ${HOME}/gammapy-fermi-lat-data
