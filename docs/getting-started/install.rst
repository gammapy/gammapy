.. include:: ../references.txt

.. _install:

Installation
============

Using Anaconda
--------------
The easiest way to install Gammapy is to install the Anaconda distribution.
The installation is explained in the :ref:`quickstart` section.

.. _virtual-envs:

Using virtual environments
--------------------------

We recommend to create an isolated virtual environment for each version of Gammapy, so that you have full
control over additional packages that you may use in your analysis. We provide, for each stable release of Gammapy,
a YAML file that allows you to easily create a specific conda execution environment. This could also help you on
improving reproducibility within the users community. See installation instructions on :ref:`quickstart` section.

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
