
.. _virtual-envs:

Virtual Environments
====================

We recommend to create an isolated virtual environment for each version of Gammapy, so that you have full
control over additional packages that you may use in your analysis. This will also help you on improving
reproducibility within the user community.

Conda Environments
------------------
For convenience we also provide, for each stable release of Gammapy,
a YAML file that allows you to easily create a specific conda execution environment.
See  :ref:`start installation quick instructions <quickstart-setup>` section. To create a new custom
environment for your analysis with conda you can use:

.. code-block:: bash

    $ conda env create -n my-gammapy-analysis

And activate it:

.. code-block:: bash

    $ conda activate my-gammapy-analysis

After that you can install Gammapy using `conda` / `mamba` as well as other packages you may need.

.. code-block:: bash

    $ conda install gammapy ipython jupyter

To leave the environment, you may activate another one or just type:

.. code-block:: bash

    $ conda deactivate

If you want to remove an virtual environment again you can use the command below:

.. code-block:: bash

    $ conda env remove -n my-gammapy-analysis

It also recommended to create a custom `environment.yaml` file, which lists all the dependencies and
additional packages you like to use explicitly. More detailed instructions on how to work with
conda environments you can find in the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__.


Venv Environments
-----------------

You may prefer to create your virtual environments with Python `venv` command instead of using Anaconda.
To create a virtual environment with `venv` (Python 3.5+ required) run the command:

.. code-block:: bash

    $ python -m venv my-gammapy-analysis

which will create one in a `my-gammapy-analysis` folder. To activate it:

.. code-block:: bash

    $ ./my-gammapy-analysis/bin/activate

After that you can install Gammapy using `pip` as well as other packages you may need.

.. code-block:: bash

    $ pip install gammapy ipython jupyter

To leave the environment, you may activate another one or just type:

.. code-block:: bash

    $ deactivate

More detailed instructions on how to work with virtual environments you can find in the `Python documentation <https://docs.python.org/3/library/venv.html>`__.
