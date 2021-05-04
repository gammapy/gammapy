.. include:: ../references.txt

.. _install:

Installation
============

Using Anaconda
--------------
The easiest and recommended way to install Gammapy is to install the Anaconda distribution.
The installation is explained in the :ref:`quickstart` section.

.. _versioned-envs:

Using versioned environments
----------------------------

We also provide an environment definition file, so you can get additional useful packages together
with gammapy in an isolated environment:

.. code-block:: bash

    curl -O https://gammapy.org/download/install/gammapy-0.18.2-environment.yml
    conda env create -f gammapy-0.18.2-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.

This creates a conda environment called ``gammapy-0.18.2`` which you can activate via:

.. code-block:: bash

    conda activate gammapy-0.18.2

Note that you have to execute that activation command (but not the environment
creation command) every time you open a new terminal window, because the default
with conda is the base environment, which does not have Gammapy installed.

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
