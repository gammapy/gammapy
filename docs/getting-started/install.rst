.. include:: ../references.txt

.. _installation:

Installation
============

.. _anaconda:

Using Anaconda / Miniconda
--------------------------

The easiest way to install Gammapy is to install the Anaconda
distribution from https://www.anaconda.com/download/ and then to install
Gammapy and its dependencies by executing this command in a terminal:

.. code-block:: bash

    conda install -c conda-forge gammapy

Though this is one line command is the standard way to install a software package using Anaconda, **we recommend to
make use of an environment definition file** that we provide, so you can get additional useful packages together with
Gammapy in a virtual isolated environment. If you want to learn about using virtual environments see
:ref:`virtual-envs`. In order to proceed in this way, just copy and paste in your terminal the two lines below:

.. code-block:: bash

    curl -O https://gammapy.org/download/install/gammapy-0.19-environment.yml
    conda env create -f gammapy-0.19-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.


Using Mamba
-----------
Alternatively, you can use `mamba <https://mamba.readthedocs.io/>` for the installation.
Mamba is an alternative package manager that support most of conda’s command but offers higher installation
speed and more reliable environment solutions. To install ``mamba`` in the base environment:

.. code-block:: bash

    conda install mamba -n base -c conda-forge

then:

.. code-block:: bash

    curl -O https://gammapy.org/download/install/gammapy-0.19-environment.yml
    mamba env create -f gammapy-0.19-environment.yml


Both options will create a conda environment called ``gammapy-0.19`` which you can activate via:

.. code-block:: bash

    conda activate gammapy-0.19

Note that you have to execute that activation command (but not the environment
creation command) every time you open a new terminal window, because the default
with conda is the base environment, which might not have Gammapy installed.

To check your Gammapy installation, you can use this command:

.. code-block:: bash

    gammapy info

To leave the environment, you may activate another one or just type:

.. code-block:: bash

    conda deactivate

.. _virtual-envs:

Using virtual environments
--------------------------

We recommend to create an isolated virtual environment for each version of Gammapy, so that you have full
control over additional packages that you may use in your analysis. We provide, for each stable release of Gammapy,
a YAML file that allows you to easily create a specific conda execution environment. This could also help you on
improving reproducibility within the users community. See installation instructions on :ref:`getting-started` section.

You may prefer to create your virtual environments with Python `venv` command instead of using Anaconda.
To create a virtual environment with `venv` (Python 3.5+ required) run the command:

.. code-block:: bash

    $ python -m venv gammapy-env

which will create one in a `gammapy-env` folder. To activate it:

.. code-block:: bash

    $ . gammapy-env/bin/activate

After that you can install Gammapy using `pip` as well as other packages you may need.

To leave the environment, you may activate another one or just type:

.. code-block:: bash

    $ deactivate

.. _install-pip:

Using pip
---------

To install the latest Gammapy **stable** version (see `Gammapy page on PyPI`_)
using `pip`_:

.. code-block:: bash

   $ python -m pip install gammapy

To install the current Gammapy **development** version using `pip`_:

.. code-block:: bash

   $ python -m pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

Or like this, if you want to study or edit the code locally:

.. code-block:: bash

   $ git clone https://github.com/gammapy/gammapy.git
   $ cd gammapy
   $ pip install .

How to get set up for Gammapy development is described here: :ref:`dev_setup`

.. _install-other:

Using other package managers
----------------------------

Gammapy has been packaged for some Linux package managers. E.g. on Debian, you
can install Gammapy via:

.. code-block:: bash

    sudo apt-get install python3-gammapy

To get a more fully featured scientific Python environment, you can install
other Python packages using the system package manager (``apt-get`` in this
example), and then on top of this install more Python packages using ``pip``.

Example:

.. code-block:: bash

    sudo apt-get install \
        python3-pip python3-matplotlib \
        ipython3-notebook python3-gammapy

    python3 -m pip install antigravity

Note that also on Linux, the recommended way to install Gammapy is via
``conda``. The ``conda`` install is maintained by the Gammapy team and gives you
usually a very recent version (releases every 2 months), whereas the Linux
package managers typically have release cycles of 6 months, or yearly or longer,
meaning that you'll get an older version of Gammapy. But you can always get a
recent version via pip:

.. code-block:: bash

    sudo apt-get install python3-gammapy
    pip install -U gammapy

Upgrade existing installation
=============================

Using Anaconda / Miniconda
--------------------------

We recommend to make use of a **conda environment definition file** that we provide for each version
of Gammapy, so you can get a specific version of Gammapy as we as its library dependencies and additional
useful packages in a virtual isolated environment.

You may find below the commands used to upgrade Gammapy to the v0.19 version.

.. code-block:: bash

    $ curl -O https://gammapy.org/download/install/gammapy-0.19-environment.yml
    $ conda env create -f gammapy-0.19-environment.yml


If you want to remove a previous version og Gammapy just use the standard conda command below.

.. code-block:: bash

    $ conda env remove -n gammapy-0.18.2

In case you are working with the development version environment and you want to update this
environment with the content present in `environment-dev.yml` see below.

.. code-block:: bash

    $ conda env update environment-dev.yml --prune
