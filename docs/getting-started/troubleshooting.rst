.. include:: ../references.txt

.. _troubleshooting:


Troubleshooting
===============


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


Help!?
------

If you have any questions or issues, please ask for help on the Gammapy Slack,
mailing list or on Github, see `Gammapy contact`_.
