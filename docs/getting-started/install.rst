.. include:: ../references.txt

.. _installation:

Installation
============

There are numerous ways to install Python and Gammapy as a user.
On this page, we list the most common ones.
In general, **we recommend using** :ref:`virtual environments <virtual-envs>` or
:ref:`project-based installation <install-project-based>` when using Gammapy.
Specifically with the latter, you have complete control over the
additional packages that you may use in your analysis and you work with
well defined computing environments. This enables you to easily share your
work and ensure **reproducibility of your scientific analysis results**.
You can also :ref:`install Gammapy for development <dev_setup>`.


.. _environment-based-installation:

Environment-based
-----------------

Using Conda / Mamba
^^^^^^^^^^^^^^^^^^^
The easiest way to install Gammapy is to use one of the conda
Python distribution software (see the `conda installation guide`_).
`Miniforge`_ is the recommended way to use conda/mamba with the
``conda-forge`` channel.

To install the latest stable version of Gammapy and its dependencies using conda,
execute this command in a terminal:

.. code-block:: bash

    conda install -c conda-forge gammapy

To update an existing installation you can use:

.. code-block:: bash

    conda update gammapy

To install a specific version of Gammapy just execute:

.. code-block:: bash

    conda install -c conda-forge gammapy=2.0

If you encounter any issues you can check the :ref:`troubleshoot` guide.

Alternatively, you can use `Mamba`_, an alternative package manager
that supports most of conda's commands but offers higher installation
speed and more reliable environment solutions. Just simply replace the
``conda`` with the ``mamba`` in the commands above. Miniforge also
comes with mamba pre-installed, so you can use it right away.

Using pip
^^^^^^^^^

To install the latest Gammapy **stable** version (see `Gammapy page on PyPI`_)
using `pip`_:

.. code-block:: bash

   python -m pip install gammapy

This will install Gammapy with the required dependencies only.

To install Gammapy with all optional dependencies, you can specify:

.. code-block:: bash

   python -m pip install gammapy[all]


To update an existing installation use:

.. code-block:: bash

   python -m pip install gammapy --upgrade

To install a specific version of Gammapy use:

.. code-block:: bash

    python -m pip install gammapy==2.0

To install the current Gammapy **development** version with `pip`_ use:

.. code-block:: bash

   python -m pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

If you want to study or edit the code locally, use the following:

.. code-block:: bash

   git clone https://github.com/gammapy/gammapy.git
   cd gammapy
   python -m pip install .

If you encounter any issues you can check the :ref:`troubleshoot` guide.

.. note::
   This will install Gammapy in the current Python environment. If
   you want to install it in a specific environment, make sure to
   activate it first and then execute the above command.

.. warning::
   Note that environment-based installs do not automatically document
   your full dependency state, which can make analyses harder to
   reproduce later. For full reproducibility, we recommend
   :ref:`install-project-based` installations.


.. _install-project-based:

Project-based
-------------

Project-based environments are the best way to manage your analysis projects.

* Faster installation method than pip and conda/mamba.
* They create an isolated environment that is tightly coupled to a
  single analysis project.
* Fully reproducible environments and analysis thanks to a lock file
  that pins every dependency to exact versions, so anyone cloning the
  project gets an identical environment.
* Makes projects easy to share and set up consistently across systems.
* Follows best practices for reproducible science.

Using uv
^^^^^^^^
`uv`_ is a modern Python package manager that provides a simple and
efficient way to create and manage project-based environments (only
with Python dependencies).

1. Install **uv** following the `uv installation instructions`_.

2. Create a new project in a new directory, by executing the following in a terminal:

   .. code-block:: bash

      uv init my-gammapy-analysis
      cd my-gammapy-analysis

3. Then, to install Gammapy, execute this command in a terminal:

   .. code-block:: bash

      uv add gammapy

This will create a set of files in the new directory *under version control*,
including a ``pyproject.toml`` file and a ``uv.lock`` file that contains the
exact versions of all used dependencies. Further dependencies can added be
via ``uv add`` command again and they will be added to the lock file. To update
Gammapy, execute ``uv add gammapy`` again and it will update to the latest version.
You can also install a specific version like, for example, ``uv add gammapy==2.1``.

Using pixi
^^^^^^^^^^
`pixi`_ is a similar package manager that also provides project-based
environments in a fast and reproducible way. The main difference is that pixi also
supports non-Python dependencies, which can be useful for some users. With **pixi**,
you can also define tasks that can be executed in the environment.

1. Install **pixi** following the `pixi installation instructions`_.

2. Create a new project in a new directory, by executing the following in a terminal:

   .. code-block:: bash

      pixi init my-second-gammapy-analysis
      cd my-second-gammapy-analysis

3. Then, to install Gammapy, execute this command in a terminal:

   .. code-block:: bash

      pixi add gammapy

Find more details on setting up a workspace in the `pixi documentation`_.

.. _install-other:

Run your analysis in the project-based environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   All the following commands are similar if you use **pixi** instead of **uv**.

Inside the `uv` directory created previously, you can run a Python analysis
script through:

.. code-block:: bash

   uv run python my-analysis-script.py

Working with Jupyter notebook
""""""""""""""""""""""""""""""
First install jupyterlab inside your `uv` project:

.. code-block:: bash

     uv add jupyterlab

Then, start jupyterlab with:

.. code-block:: bash

   uv run jupyter lab

Using other package managers
----------------------------

Gammapy has been packaged for some Linux package managers. E.g. on Debian, you
can install Gammapy via:

.. code-block:: bash

    sudo apt-get install python3-gammapy

To get a more fully featured scientific Python environment, you can install
other Python packages using the system package manager (``apt-get`` in this
example), and then on top of this install more Python packages using ``pip``.


For example:

.. code-block:: bash

    sudo apt-get install \
        python3-pip python3-matplotlib \
        ipython3-notebook python3-gammapy

    python3 -m pip install gammapy

Note that the Linux package managers typically have release cycles of 6 months,
or yearly or longer, meaning that you'll typically  get an older version of
Gammapy. However, you can always get the recent version via
:ref:`pip or conda <environment-based-installation>`.
