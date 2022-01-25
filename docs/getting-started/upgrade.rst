.. _upgrade:

Upgrade
=======

We recommend to make use of a **conda environment definition file** that we provide for each version
of Gammapy, so you can get a specific version of Gammapy as we as its library dependencies and additional
useful packages in a virtual isolated environment.

You may find below the commands used to upgrade Gammapy to the v0.19 version.

.. code-block:: bash

    $ curl -O https://gammapy.org/download/install/gammapy-0.19-environment.yml
    $ conda env create -f gammapy-0.19-environment.yml


If you want to remove a previous version og Gammapy just use the standard conda command below.

.. code-block:: bash

    $ conda env remove -n gammapy-0.18.2

In case you are working with the development version environment and you want to update this
environment with the content present in `environment-dev.yml` see below.

.. code-block:: bash

    $ conda env update environment-dev.yml --prune