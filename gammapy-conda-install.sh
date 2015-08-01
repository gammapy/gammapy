#!/usr/bin/env bash

#
# A bash script to quickly install Gammapy from scratch using miniconda.
#
# This is a binary install, so the install speed is mostly limited by
# download speed (typically a few minutes).
#
#     bash "$(curl -fsSL https://raw.githubusercontent.com/gammapy/gammapy/master/gammapy-conda-install.sh)"
#

export INSTALL_DIR="$PWD/miniconda"

# Detect platform: http://stackoverflow.com/a/17072017/498873
if [ "$(uname)" == "Darwin" ]; then
    export PLATFORM="MacOSX"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    export PLATFORM="Linux"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    export PLATFORM="Windows"
fi

export PLATFORM="Linux"

wget http://repo.continuum.io/miniconda/Miniconda3-latest-$(PLATFORM)-x86_64.sh -O miniconda.sh

bash miniconda.sh -b -p $(INSTALL_DIR)
export PATH="$(INSTALL_DIR)/bin:$PATH"
conda config --set always_yes yes --set changeps1 no

conda config --add channels astropy
conda config --add channels sherpa

conda update -q conda

conda install pip scipy matplotlib scikit-image scikit-learn astropy h5py pandas ipython-notebook
conda install reproject aplpy wcsaxes naima astroplan gwcs photutils
conda install sherpa
conda install gammapy
