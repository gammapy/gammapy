# This is the Dockerfile to run Gammapy on Binder
#

FROM continuumio/miniconda3
MAINTAINER Gammapy developers <gammapy@googlegroups.com>

# compilers
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip 

# install good version of notebook for Binder
# RUN pip install --no-cache-dir notebook==5.*

# install dependencies - including the dev version of Gammapy
COPY environment.yml binder.py tmp/
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

# copy repo in /home/gammapy
# COPY . ${HOME}

RUN gammapy download --release=master --dest=${HOME}/gammapy-tutorials notebooks

# remove tutorials using CTA 1DC
RUN rm ${HOME}/gammapy-tutorials/notebooks-master/analysis_3d.ipynb
RUN rm ${HOME}/gammapy-tutorials/notebooks-master/cta_1dc_introduction.ipynb
RUN rm ${HOME}/gammapy-tutorials/notebooks-master/cta_data_analysis.ipynb
RUN rm ${HOME}/gammapy-tutorials/notebooks-master/simulate_3d.ipynb
RUN rm ${HOME}/gammapy-tutorials/notebooks-master/spectrum_simulation_cta.ipynb

# setting ownerships
USER root
RUN chown -R ${NB_UID} ${HOME}

# start JupyterLab server in tutorials dir
USER ${NB_USER}
WORKDIR ${HOME}/gammapy-tutorials/notebooks-master

# env vars used in tutorials
RUN git clone https://github.com/gammapy/gammapy-extra.git ${HOME}/gammapy-tutorials/gammapy-extra
ENV GAMMAPY_EXTRA ${HOME}/gammapy-tutorials/gammapy-extra

RUN git clone https://github.com/gammapy/gamma-cat.git  ${HOME}/gammapy-tutorials/gammapy-cat
ENV GAMMAPY_CAT ${HOME}/gammapy-tutorials/gammapy-cat

RUN git clone https://github.com/gammapy/gammapy-fermi-lat-data.git ${HOME}/gammapy-tutorials/gammapy-fermi-lat-data
ENV GAMMAPY_FERMI_LAT_DATA ${HOME}/gammapy-tutorials/gammapy-fermi-lat-data
