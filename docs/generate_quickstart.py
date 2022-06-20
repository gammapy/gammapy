# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Builds quickstart section for Gammapy documentation
import sys
#from gammapy import __version__ as version

template_quickstart = """
.. _quickstart-setup:

Quickstart Setup
----------------

The best way to get started and learn Gammapy are the :ref:`tutorials`. For
convenience we provide a pre-defined conda environment file, so you can
get additional useful packages together with Gammapy in a virtual isolated
environment. First install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__
and then just execute the following commands in the terminal:

.. code-block:: bash

    $ curl -O https://gammapy.org/download/install/gammapy-version-environment.yml
    $ conda env create -f gammapy-version-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.

Once the environment has been created you can activate it using:

.. code-block:: bash

    $ conda activate gammapy-version
"""

template_download = """
.. _download-datasets:

You can now proceed to download the Gammapy tutorial notebooks and the example
datasets. The total size to download is ~180 MB. Select the location where you
want to install the datasets and proceed with the following commands:

.. code-block:: bash

    $ gammapy download notebooks --release version
    $ gammapy download datasets
    $ export GAMMAPY_DATA=$PWD/gammapy-datasets
    
ou might want to put the definition of the ``$GAMMAPY_DATA`` environment
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
    
"""

def replace_version(template_str, version):
    new_content = template_str.replace("version", version)
    return new_content

def main():
    version = sys.argv[1]
    fp = open('getting-started/quickstart.inc', 'w')
    if "dev" not in version:
        fp.write(replace_version(template_quickstart, version))

    fp = open('getting-started/download.inc', 'w')
    if "dev" not in version:
        fp.write(replace_version(template_download, version))

if __name__ == "__main__":
    main()