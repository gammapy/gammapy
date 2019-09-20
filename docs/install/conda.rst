.. include:: ../references.txt

.. _install-conda:

Gammapy conda installation
==========================


If you want to use some Python package together with Gammapy that is not
pre-installed as part of that conda environment, you can use ``conda`` or
``pip`` to install it. To give an example: you could use ``conda install
pandas`` or ``pip install astroplan``.

Background information
======================

Anaconda is a free scientific Python distribution available on Linux, MacOS and
Windows. By default, it is installed into your home directory (or a directory of
your choice), so you can install (or easily remove) it on any computer you have
access to.

Another big advantage of conda is that is is a binary package manager. This
means that no C, C++ or Fortran compiler is needed on your machine, and
installation is very fast. Usually it takes about a Gigabyte of disk space, and
installation takes a few minutes, the speed depends on your internet download
and hard disk write speed.

When installing Anaconda, you get the `conda`_ command line tool, which is a
package and environment manager. You can install any number of environments, and
within a given environment, you can install any number of packages and versions
of your choosing. When a new stable version of Gammapy comes out, and you follow
these install instructions for that new version again, and you get a new
separate environment and can use either the old or the new on, as you like.

What the commands above do is to create a dedicated conda environment for
Gammapy, and to install all required and many optional dependencies for Gammapy,
with known good versions that we have extensively tested on Linux, MacOS and
Windows. You get a reproducible execution environment, anyone installing that
environment should get almost identical results on any machine. There are tiny
differences (e.g. 0.0001% in flux) due to different floating point precision of
the underlying numerics libraries and compilers used for different platforms,
especially for the Numpy dependency.



To install the latest Gammapy **stable** version as well as the most common
optional dependencies for Gammapy, first install `Anaconda
<http://continuum.io/downloads>`__ and then run this commands:

.. code-block:: bash

    conda config --add channels conda-forge --add channels sherpa
    conda install gammapy naima sherpa \
        scipy matplotlib ipython-notebook \
        cython click reproject iminuit

To update to the latest version:

.. code-block:: bash

    conda update --all
    conda update gammapy

Overall ``conda`` is a great cross-platform package manager, you can quickly
learn how to use it by reading the `conda docs
<http://conda.pydata.org/docs/>`__.


Stable version
--------------

You can install the latest stable version of Gammapy with conda::

    conda install -c conda-forge gammapy

or with pip::

    python -m pip install gammapy

Gammapy is not yet available in the Linux distributions, i.e. at this time you
can't install it with e.g. ``apt-get`` or ``yum``.
