.. _quickstart-setup:

Quickstart Setup
----------------

The best way to get started and learn Gammapy are the :ref:`tutorials`. For
convenience we provide a pre-defined conda environment file, so you can
get additional useful packages together with Gammapy in a virtual isolated
environment. First install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
and then just execute the following commands in the terminal:

.. substitution-code-block:: bash

    $ curl -O https://gammapy.org/download/install/gammapy-|release|-environment.yml
    $ conda env create -f gammapy-|release|-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.

.. note::

    For Apple silicon M1 (`arm64`) architectures you also have to open the
    environment file and delete the `sherpa` entry, as currently there are
    no conda packages available. However you can later install `sherpa`
    in the environment using `python -m pip install sherpa`.

Once the environment has been created you can activate it using:

.. substitution-code-block:: bash

    $ conda activate gammapy-|release|

You can now proceed to download the Gammapy tutorial notebooks and the example
datasets. The total size to download is ~180 MB. Select the location where you
want to install the datasets and proceed with the following commands:

.. substitution-code-block:: bash

    $ gammapy download notebooks
    $ gammapy download datasets
    $ export GAMMAPY_DATA=$PWD/gammapy-datasets/|release|

You might want to put the definition of the ``$GAMMAPY_DATA`` environment
variable in your shell profile setup file that is executed when you open a new
terminal (for example ``$HOME/.bash_profile``).

.. note::

    If you are not using the ``bash`` shell, handling of shell environment variables
    might be different, e.g. in some shells the command to use is ``set`` or something
    else instead of ``export``, and also the profile setup file will be different.
    On Windows, you should set the ``GAMMAPY_DATA`` environment variable in the
    "Environment Variables" settings dialog, as explained e.g.
    `here <https://docs.python.org/3/using/windows.html#excursus-setting-environment-variables>`__

Finally start a notebook server by executing:

.. code-block:: bash

    $ cd notebooks
    $ jupyter notebook

If you are new to conda, Python and Jupyter, maybe also read the :ref:`using-gammapy` guide.
If you encountered any issues you can check the :ref:`troubleshoot` guide.
