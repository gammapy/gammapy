.. include:: ../references.txt

.. _installation:

Installation
============

There are many ways to install Python and Gammapy as a user. On this page we list the most common ones.
In general **we recommend to use virtual environments** when using Gammapy. This ways you have full
control over additional packages that you may use in your analysis and you work with
well defined computing environments, that you can share and allow **reproducibility of your
scientific analysis results**. If you want to learn about using virtual environments see :ref:`virtual-envs`.
You can also :ref:`install Gammapy for development <dev_setup>`.

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

    $ conda install -c conda-forge gammapy=0.19

If you encountered any issues you can check the :ref:`troubleshoot` guide.

Using Mamba
-----------
Alternatively, you can use `Mamba <https://mamba.readthedocs.io/>`__ for the installation.
Mamba is an alternative package manager that support most of conda’s command but offers higher installation
speed and more reliable environment solutions. To install ``mamba`` in the Conda base environment:

.. code-block:: bash

    $ conda install mamba -n base -c conda-forge

then:

.. code-block:: bash

    $ mamba install gammapy

Mamba supports of the commands that are available for conda. So updating and installing specific versions
works the same way as above except for replacing the ``conda`` with the ``mamba`` command.

.. _install-pip:

Using pip
---------

To install the latest Gammapy **stable** version (see `Gammapy page on PyPI`_)
using `pip`_:

.. code-block:: bash

   $ python -m pip install gammapy

To update an existing installation you can use:

.. code-block:: bash

   $ python -m pip install gammapy --upgrade

To install a specific version of Gammapy you can use:

.. code-block:: bash

    $ python -m pip install gammapy==0.19

To install the current Gammapy **development** version using `pip`_ you can use:

.. code-block:: bash

   $ python -m pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

Or like this, if you want to study or edit the code locally:

.. code-block:: bash

   $ git clone https://github.com/gammapy/gammapy.git
   $ cd gammapy
   $ python -m pip install .

If you encountered any issues you can check the :ref:`troubleshoot` guide.

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

    $ python3 -m pip install antigravity

Note that the Linux package managers typically have release cycles of 6 months,
or yearly or longer, meaning that you'll typically  get an older version of
Gammapy. But you can always get a recent version via `pip` or `conda` (see above).
