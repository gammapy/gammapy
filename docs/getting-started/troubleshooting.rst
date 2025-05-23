.. include:: ../references.txt

.. _troubleshoot:


Troubleshooting
===============


Check your setup
----------------
To access detailed information about your Gammapy installation, you can execute the following
command. It will provide insight into the installation directly to the terminal.

.. code-block:: bash

    gammapy info

If you encounter some issues, the following commands can help you in troubleshooting
your setup:

.. code-block:: bash

    conda info
    which python
    which ipython
    which jupyter
    which gammapy
    env | grep PATH
    python -c 'import gammapy; print(gammapy); print(gammapy.__version__)'

You can also use the following commands to check which conda environment is active and to
list all available environments:

.. code-block:: bash

    conda info
    conda env list

For those that are new to conda, you can consult the `conda cheat sheet`_, which
lists the common commands for installing packages and working with environments.


Install issues
--------------

If you're experiencing issues and believe you are using the incorrect Python or Gammapy version, or
if you encounter problems importing Gammapy, you should check your Python executable and import path
to help you resolve the issue:

.. code-block:: python

    import sys
    print(sys.executable)
    print(sys.path)

To check your Gammapy version use the following commands:

.. code-block:: python

    import gammapy
    print(gammapy)
    print(gammapy.__version__)


You should now be ready to start using Gammapy. Let's move on to the
:ref:`tutorials`.


Help!?
------

If you have any questions or issues, please ask for help on the Gammapy Slack,
mailing list or on GitHub (see `Gammapy contact`_).
