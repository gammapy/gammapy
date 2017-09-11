#!/usr/bin/env bash

#
# A bash script to quickly install Gammapy from scratch using miniconda.
#
# This is a binary install, so the install speed is mostly limited by
# download speed (typically ~ 1 gigabyte and a few minutes)
#
# To download and execute in one command:
#
#     bash "$(curl -fsSL https://raw.githubusercontent.com/gammapy/gammapy/master/gammapy-conda-install.sh)"
#

set -x

INSTALL_DIR="$PWD/miniconda"
echo "INSTALL_DIR = $INSTALL_DIR"

# Detect platform: http://stackoverflow.com/a/394247/498873
if [ "$(uname)" == "Darwin" ]; then
    PLATFORM="MacOSX"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    PLATFORM="Linux"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    PLATFORM="Windows"
fi
echo "PLATFORM = $PLATFORM"

wget http://repo.continuum.io/miniconda/Miniconda3-latest-$PLATFORM-x86_64.sh -O miniconda.sh

bash miniconda.sh -b -p $INSTALL_DIR
export PATH="$INSTALL_DIR/bin:$PATH"
conda config --set always_yes yes --set changeps1 no

conda config --add channels conda-forge
conda config --add channels sherpa

conda update -q conda
# Disk space now: 140 MB

# Finally ... install Gammapy and the most useful dependencies
conda install gammapy naima \
    iminuit scipy matplotlib ipython-notebook
# Disk space now: 200 MB

# Nice to have extras
conda install \
    scikit-image scikit-learn h5py pandas \
    aplpy photutils
# Disk space now: 747 MB

pip install reproject

conda install sherpa
