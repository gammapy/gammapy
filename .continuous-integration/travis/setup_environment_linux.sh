#!/bin/bash

# Install conda
wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
export PATH=/Users/travis/miniconda2/bin:$PATH
conda update --yes conda

# Install Python dependencies
source "$( dirname "${BASH_SOURCE[0]}" )"/setup_dependencies_common.sh


# Make matplotlib testing work on travis-ci
export DISPLAY=:99.0
sh -e /etc/init.d/xvfb start
export QT_API=pyqt
