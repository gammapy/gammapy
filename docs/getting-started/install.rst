.. include:: ../references.txt

.. _installation:

Installation
============

There are numerous ways to install Python and Gammapy as a user. On this page, we list the most common ones.
In general, **we recommend using** :ref:`virtual environments <virtual-envs>` when using Gammapy. This
way you have complete control over the additional packages that you may use in your analysis and you work with
well defined computing environments. This enables you to easily share your work and ensure **reproducibility of your
scientific analysis results**. You can also :ref:`install Gammapy for development <dev_setup>`.


.. _anaconda:

Using Anaconda / Miniconda
--------------------------
The easiest way to install Gammapy is to install the `Anaconda <https://www.anaconda.com/download/>`__
or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ Python distribution.
To install the latest stable version of Gammapy and its dependencies using conda,
execute this command in a terminal:

.. code-block:: bash

    $ conda install -c conda-forge gammapy

To update an existing installation you can use:

.. code-block:: bash

    $ conda update gammapy

To install a specific version of Gammapy just execute:

.. code-block:: bash

    $ conda install -c conda-forge gammapy=1.0

If you encounter any issues you can check the :ref:`troubleshoot` guide.

Using Mamba
-----------
Alternatively, you can use `Mamba <https://mamba.readthedocs.io/>`__ for the installation.
Mamba is an alternative package manager that supports most of conda’s commands but offers higher installation
speed and more reliable environment solutions. To install ``mamba`` in the Conda base environment:

.. code-block:: bash

    $ conda install mamba -n base -c conda-forge

Then install Gammapy through:

.. code-block:: bash

    $ mamba install gammapy

Mamba supports the same commands available in conda. Therefore, updating and installing specific versions
follows the same process as above, just simply replace the ``conda`` command with the ``mamba`` command.

.. _install-pip:

Using pip
---------

To install the latest Gammapy **stable** version (see `Gammapy page on PyPI`_)
using `pip`_:

.. code-block:: bash

   $ python -m pip install gammapy

This will install Gammapy with the required dependencies only.
    
To install Gammapy with all optional dependencies, you can specify:

.. code-block:: bash

   $ python -m pip install gammapy[all]


To update an existing installation use:

.. code-block:: bash

   $ python -m pip install gammapy --upgrade

To install a specific version of Gammapy use:

.. code-block:: bash

    $ python -m pip install gammapy==1.0

To install the current Gammapy **development** version with `pip`_ use:

.. code-block:: bash

   $ python -m pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

If you want to study or edit the code locally, use the following:

.. code-block:: bash

   $ git clone https://github.com/gammapy/gammapy.git
   $ cd gammapy
   $ python -m pip install .

If you encounter any issues you can check the :ref:`troubleshoot` guide.

.. _install-other:

Using other package managers
----------------------------

Gammapy has been packaged for some Linux package managers. E.g. on Debian, you
can install Gammapy via:

.. code-block:: bash

    $ sudo apt-get install python3-gammapy

To get a more fully featured scientific Python environment, you can install
other Python packages using the system package manager (``apt-get`` in this
example), and then on top of this install more Python packages using ``pip``.

Example:

.. code-block:: bash

    $ sudo apt-get install \
        python3-pip python3-matplotlib \
        ipython3-notebook python3-gammapy

    $ python3 -m pip install gammapy

Note that the Linux package managers typically have release cycles of 6 months,
or yearly or longer, meaning that you'll typically  get an older version of
Gammapy. However, you can always get the recent version via `pip` or `conda` (see above).
