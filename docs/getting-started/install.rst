.. include:: ../references.txt

.. _install:

Installation
============

Using Anaconda
--------------
The easiest and recommended way to install Gammapy is to install the Anaconda distribution.
The installation is explained in the :ref:`getting_started` section.

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

Check your setup
----------------
You might want to display some info about Gammapy installed. You can execute
the following command, and it should print detailed information about your
installation to the terminal:

.. code-block:: bash

    gammapy info

If there is some issue, the following commands could help you to figure out
your setup:

.. code-block:: bash

    conda info
    which python
    which ipython
    which jupyter
    which gammapy
    env | grep PATH
    python -c 'import gammapy; print(gammapy); print(gammapy.__version__)'

You can also use the following commands to check which conda environment is active and which
ones you have set up:

.. code-block:: bash

    conda info
    conda env list

If you're new to conda, you could also print out the `conda cheat sheet`_, which
lists the common commands to install packages and work with environments.


Install issues
--------------

If you have problems and think you might not be using the right Python or
importing Gammapy isn't working or giving you the right version, checking your
Python executable and import path might help you find the issue:

.. code-block:: python

    import sys
    print(sys.executable)
    print(sys.path)

To check which Gammapy you are using you can use this:

.. code-block:: python

    import gammapy
    print(gammapy)
    print(gammapy.__version__)

Now you should be all set and to use Gammapy. Let's move on to the
:ref:`tutorials`.


.. Include toc hidden to avoid warnings in doc building

.. toctree::
    :hidden:

    dependencies
