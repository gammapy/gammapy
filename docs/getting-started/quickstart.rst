.. _quickstart-setup:

Recommended Setup
-----------------

We recommend using :ref:`virtual environments <virtual-envs>`, to do so
execute the following commands in the terminal:

.. substitution-code-block:: bash

    $ curl -O https://gammapy.org/download/install/gammapy-|release|-environment.yml
    $ conda env create -f gammapy-|release|-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.

.. note::

    For gammapy versions **prior to v1.2**, if you're using
    Apple silicon M1 (`arm64`) architectures you have to open the
    environment file and delete the `sherpa` entry. Currently there are
    no conda packages available. You can then install `sherpa`
    in the environment using `python -m pip install sherpa`.


The best way to get started and learn Gammapy are the :ref:`tutorials`.
You can download the Gammapy tutorial notebooks and the example
datasets. The total size to download is ~180 MB. Select the location where you
want to install the datasets and proceed with the following commands:

.. substitution-code-block:: bash

    $ conda activate gammapy-|release|
    $ gammapy download notebooks
    $ gammapy download datasets
    $ conda env config vars set GAMMAPY_DATA=$PWD/gammapy-datasets/|release|
    $ conda activate gammapy-|release|


The last conda commands will define the environment variable within the conda environment.
Conversely, you might want to define the ``$GAMMAPY_DATA`` environment
variable directly in your shell with:

.. substitution-code-block:: bash

    $ export GAMMAPY_DATA=$PWD/gammapy-datasets/|release|

.. note::

    If you are not using the ``bash`` shell, handling of shell environment variables
    might be different, e.g. in some shells the command to use is ``set`` or something
    else instead of ``export``, and also the profile setup file will be different.
    On Windows, you should set the ``GAMMAPY_DATA`` environment variable in the
    "Environment Variables" settings dialog, as explained e.g.
    `here <https://docs.python.org/3/using/windows.html#excursus-setting-environment-variables>`__.

Finally start a notebook server by executing:

.. code-block:: bash

    $ cd notebooks
    $ jupyter notebook

If you are new to conda, Python and Jupyter, it is recommended to also read the :ref:`using-gammapy` guide.
If you encounter any issues you can check the :ref:`troubleshoot` guide.
