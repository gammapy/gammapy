#!/bin/bash
# This is what travis-ci does for a build of the master branch
export OWNER=gammapy
export REPO=gammapy
export BRANCH=master
export SHA=39caf3bf78c877629bd6bd7513a272f76c8e1ca2
git clone --depth=50 --branch=${BRANCH} http://github.com/${OWNER}/${REPO}.git ${OWNER}/${REPO}
cd ${OWNER}/${REPO}
git checkout -qf ${SHA}
git submodule init
git submodule update

# Define environment variables
export NUMPY_VERSION=1.9
export ASTROPY_VERSION=development
export CONDA_INSTALL='conda install -c astropy-ci-extras --yes'
export PIP_INSTALL='pip install'
export ASTROPY_VERSION=development
export SETUP_CMD='test'

# Execute commands
source ~/virtualenv/python3.3/bin/activate
python --version
pip --version
export PYTHONIOENCODING=UTF8
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda
sudo apt-get update
if [[ $SETUP_CMD == build_docs* ]]; then sudo apt-get install graphviz texlive-latex-extra dvipng; fi
if [[ $TRAVIS_PYTHON_VERSION == 2.7 ]]; then export TRAVIS_PYTHON_VERSION=2.7.8; fi
conda create --yes -n test -c astropy-ci-extras python=$TRAVIS_PYTHON_VERSION
source activate test
export CONDA_INSTALL="conda install -c astropy-ci-extras --yes python=$TRAVIS_PYTHON_VERSION numpy=$NUMPY_VERSION"
if [[ $SETUP_CMD != egg_info ]]; then $CONDA_INSTALL numpy=$NUMPY_VERSION pytest pip Cython jinja2; fi
if [[ $SETUP_CMD != egg_info ]]; then $PIP_INSTALL pytest-xdist; fi
if [[ $SETUP_CMD != egg_info ]] && [[ $ASTROPY_VERSION == development ]]; then $PIP_INSTALL git+http://github.com/astropy/astropy.git#egg=astropy; fi
if [[ $SETUP_CMD != egg_info ]] && [[ $ASTROPY_VERSION == stable ]]; then $CONDA_INSTALL numpy=$NUMPY_VERSION astropy; fi
if [[ $SETUP_CMD != egg_info ]]; then $CONDA_INSTALL scipy pandas; fi
if [[ $SETUP_CMD != egg_info ]]; then $PIP_INSTALL uncertainties; fi
if [[ $SETUP_CMD != egg_info ]]; then $PIP_INSTALL git+http://github.com/astrofrog/reproject.git#egg=reproject; fi
if [[ $SETUP_CMD == build_docs* ]]; then $CONDA_INSTALL numpy=$NUMPY_VERSION Sphinx matplotlib scipy; fi
if [[ $SETUP_CMD == build_docs* ]]; then $PIP_INSTALL linkchecker; fi
if [[ $SETUP_CMD == 'test --coverage' ]]; then $PIP_INSTALL coverage coveralls; fi
python setup.py $SETUP_CMD
if [[ $SETUP_CMD == build_docs* ]]; then linkchecker docs/_build/html; fi
