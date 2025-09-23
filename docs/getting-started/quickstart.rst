.. _quickstart-setup:

Recommended Setup
-----------------

We recommend using :ref:`virtual environments <virtual-envs>`, to do so
execute the following commands in the terminal:

.. substitution-code-block:: console

    curl -O https://gammapy.org/download/install/gammapy-|release|-environment.yml
    conda env create -f gammapy-|release|-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    line with ``healpy``. This is an optional dependency that
    currently isn't available on Windows.

.. note::
    To avoid some installation issues, ``sherpa`` is not part of the environment file provided. If required, you can install ``sherpa`` in your environment using ``python -m pip install sherpa``.

**The best way to get started and learn Gammapy is to understand the** :ref:`package_structure`.
You can download the Gammapy tutorial notebooks and the example
datasets. The total size to download is ~180 MB. Select the location where you
want to install the datasets and proceed with the following commands:

.. substitution-code-block:: console

    conda activate gammapy-|release|
    gammapy download notebooks
    gammapy download datasets
    conda env config vars set GAMMAPY_DATA=$PWD/gammapy-datasets/|release|
    conda activate gammapy-|release|


The last conda commands will define the environment variable within the conda environment.
Conversely, you might want to define the ``$GAMMAPY_DATA`` environment
variable directly in your shell with:

.. substitution-code-block:: console

    export GAMMAPY_DATA=$PWD/gammapy-datasets/|release|

.. note::

    If you are not using the ``bash`` shell, handling of shell environment variables
    might be different, e.g. in some shells the command to use is ``set`` or something
    else instead of ``export``, and also the profile setup file will be different.
    On Windows, you should set the ``GAMMAPY_DATA`` environment variable in the
    "Environment Variables" settings dialog, as explained e.g.
    `here <https://docs.python.org/3/using/windows.html#excursus-setting-environment-variables>`__.


To check that the gammapy environment is working, open a terminal and type:

.. substitution-code-block:: console

    conda activate gammapy-|release|
    gammapy info


Jupyter
-------
Once you have activated your gammapy environment you can start
a notebook server by executing::

    cd notebooks
    jupyter notebook


Another option is to utilise the ipykernel functionality of Jupyter Notebook, which allows you
to choose a kernel from a predefined list. To add kernels to the list, use the following
command lines:

.. substitution-code-block:: console

    conda activate gammapy-|release|
    python -m ipykernel install --user --name gammapy-|release| --display-name "gammapy-|release|"

To make use of it, simply choose it as your kernel when launching `jupyter lab` or `jupyter notebook`.

If you are new to conda, Python and Jupyter, it is recommended to also read the :ref:`using-gammapy` guide.
If you encounter any issues you can check the :ref:`troubleshoot` guide.
