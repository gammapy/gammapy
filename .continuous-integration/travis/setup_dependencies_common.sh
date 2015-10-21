#!/bin/bash -x

# CONDA
conda create --yes -n test -c astropy-ci-extras python=$PYTHON_VERSION pip
source activate test

# EGG_INFO
if [[ $SETUP_CMD == egg_info ]]
then
  return  # no more dependencies needed
fi

# PEP8
if [[ $MAIN_CMD == pep8* ]]
then
  pip install pep8
  return  # no more dependencies needed
fi

# CORE DEPENDENCIES
conda install --yes pytest Cython jinja2 psutil

# These dependencies are just needed for some functionality, they are not core.
# But the pytest runner fails with an ImportError if we don't put it here
conda install --yes click

# NUMPY
if [[ $NUMPY_VERSION == dev ]]
then
  pip install git+http://github.com/numpy/numpy.git
  export CONDA_INSTALL="conda install --yes python=$PYTHON_VERSION"
else
  conda install --yes numpy=$NUMPY_VERSION
  export CONDA_INSTALL="conda install --yes python=$PYTHON_VERSION numpy=$NUMPY_VERSION"
fi

# ASTROPY
if [[ $ASTROPY_VERSION == dev ]]
then
  pip install git+http://github.com/astropy/astropy.git
else
  $CONDA_INSTALL numpy=$NUMPY_VERSION astropy=$ASTROPY_VERSION
fi


# Now set up shortcut to conda install command to make sure the Python and Numpy
# versions are always explicitly specified.

# OPTIONAL DEPENDENCIES
if $OPTIONAL_DEPS
then
  $CONDA_INSTALL scipy h5py matplotlib pyyaml scikit-image scikit-learn pandas
  conda install --yes --channel astropy pyregion naima photutils wcsaxes
  pip install uncertainties reproject

  if [[ $PYTHON_VERSION == 2.7 ]]
  then
    conda install --yes --channel astropy numpy=$NUMPY_VERSION iminuit
    conda install --yes --channel https://conda.anaconda.org/cxc/channel/dev sherpa
  fi
fi

# DOCUMENTATION DEPENDENCIES
if [[ $SETUP_CMD == build_sphinx* ]]
then
  $CONDA_INSTALL sphinx pygments matplotlib scipy
  conda install --yes --channel astropy wcsaxes aplpy
fi

# COVERAGE DEPENDENCIES
# cpp-coveralls must be installed first.  It installs two identical
# scripts: 'cpp-coveralls' and 'coveralls'.  The latter will overwrite
# the script installed by 'coveralls', unless it's installed first.
if [[ $SETUP_CMD == 'test -V --coverage' ]]
then
  # TODO can use latest version of coverage (4.0) once
  # https://github.com/astropy/astropy/issues/4175 is addressed in
  # astropy release version.
  pip install coverage==3.7.1;
  pip install cpp-coveralls;
  pip install coveralls;
fi