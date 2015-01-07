# A docker image that's very similar to the one that
# the Gammapy build runs in on travis-ci

# *** Get a container that's similar to travis-ci

FROM ubuntu:12.04

MAINTAINER Christoph Deil

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -y wget git bzip2 gcc

WORKDIR /root

# *** These commands are very similar to the ones run on travis-ci

# Install conda
RUN wget --quiet http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
RUN chmod +x miniconda.sh
RUN ./miniconda.sh -b
ENV PATH /root/miniconda/bin:$PATH
RUN conda update --yes conda
# https://groups.google.com/a/continuum.io/d/msg/conda/HZlV2LXxWhs/yfjncv3XPSEJ
# This GCC doesn't work ... missing header limits.h error when trying to build astropy
# RUN conda install --yes -c asmeurer gcc

ENV NUMPY_VERSION 1.9
ENV ASTROPY_VERSION development
ENV TRAVIS_PYTHON_VERSION 3.4
ENV PIP_INSTALL 'pip install'
ENV SETUP_CMD 'test'

RUN conda create --yes -n test -c astropy-ci-extras python=$TRAVIS_PYTHON_VERSION

RUN source activate test
RUN conda install -y -c astropy-ci-extras --yes python=$TRAVIS_PYTHON_VERSION numpy=$NUMPY_VERSION
RUN conda install -y numpy=$NUMPY_VERSION pytest pip Cython jinja2 scipy
RUN pip install nose pytest-xdist

RUN git clone http://github.com/astropy/astropy.git
RUN cd /root/astropy && python setup.py install

RUN git clone http://github.com/gammapy/gammapy.git
RUN cd /root/gammapy && python setup.py install
