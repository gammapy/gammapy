.. include:: ../references.txt

.. _scripts:

***************************************
Command line tools  (`gammapy.scripts`)
***************************************

.. currentmodule:: gammapy.scripts

Introduction
============

Currently, Gammapy is first and foremost a Python package. This means that
to use it you have to write a Python script or Jupyter notebook, where
you import the functions and classes needed for a given analysis, and then
call them, passing parameters to configure the analysis.

In addition, we are considering options how to implement a high-level interface
for Gammapy. The two standard options for such a high-level interface are
either a bunch of command line tools (example: `Fermi ScienceTools`_), or
a config file (example: `FermiPy`_ or also HAP tool in `H.E.S.S.`_). There are
also other more fancy options for configurable analysis pipelines or systems
that let users compose analysis workflows via configuration files (e.g. YAML
or XML) instead of Python code (examples: `fact-tools`_, `ctapipe`_)
or ways to auto-expose functionality in a Python package as command line tools
(example: `python-fire`_).

Please note that the existing high-level interface in Gammapy in this
``gammapy.scripts`` package is experimental. We haven't committed to a
way to expose the available functionality via a high-level interface yet,
and only a very small subset of the functionality from Gammapy is available
in that way. Once the Gammapy Python package is a little more developed,
we probably will add a high-level interface to Gammapy. At least
for the most common functionality, advanced users will probably always
find the exiting Python interface to be the best way to use Gammapy.
Please let us know on the Gammapy mailing list what interface you would like to have!

.. _fact-tools: https://pos.sissa.it/236/865/
.. _python-fire: https://github.com/google/python-fire


.. _scripts_autodoc:

CLI
===

.. _scripts_overview:

Overview
--------

.. code-block:: text

    $ gammapy --help

      Gammapy command line interface.

      Gammapy is a Python package for gamma-ray astronomy.

      For further information, see http://gammapy.org/

    Options:
      --log-level [debug|info|warning|error]
                                      Logging verbosity level
      --ignore-warnings               Ignore warnings?
      --version                       Print version and exit
      -h, --help                      Show this message and exit.

    Commands:
      check  Run checks for Gammapy
      image  Analysis - 2D images
      info   Display information about Gammapy


Here is auto-generated documentation for all available sub-commands, arguments
and options of the ``gammapy`` command line interface (CLI).

.. click:: gammapy.scripts.main:cli
   :prog: gammapy
   :show-nested:

Implementation
==============

Currently, the command line interface (CLI) of Gammapy is implemented using `click`_
to define sub-commands, arguments and options, as well as calling the right function
that implements a given sub-command. `sphinx-click`_  is used to generate a nice
version of the help for each sub-command on this HTML page.

We have chosen to implement all functionality via a single command line tool called
``gammapy``, with each task as a subcommand like ``gammapy bin``. This

``click`` is nice and simple to use, and it's a pure Python package that we could
just bundle as a few ``.py`` files in ``gammapy.extern`` if we want to avoid
the extra external dependency. However, probably ``click`` is not the long-term
solution for Gammapy, the main features we'd like to have that the current solution
doesn't offer are:

* Support in-memory tool chain analysis pipeline. E.g. ``gammapy bin`` followed
  by ``gammapy fit`` without writing intermediate files and starting the two
  commands as separate processes.

The Gammapy command line tool uses the `setuptools entry points`_
method to automatically create command line tools when Gammapy is installed.

This means that to be able to use the tools you have to install Gammapy:

.. code-block:: bash

    $ python -m pip install .


This will install the ``gammapy-*`` wrappers in a ``bin`` folder that you need to add to your ``$PATH``,
which will then call into the appropriate function in the Gammapy package.

For Gammapy development we recommend you run this command so that you can edit
Gammapy and the tools and don't have to re-install after every change.

.. code-block:: bash

    $ python -m pip install --editable .


Most of the command line tools are implemented in the `gammapy.scripts` sub-package as thin wrappers
around functionality that's implemented in the Gammapy package as re-usable functions and classes.
In most cases all the command line tool ``main`` function does is argument passing and setting up logging.

TODO: explain somewhere that we use `sphinx-click`_ to generate cli documentation.

Write your own
==============

If you'd like to write your own command line tool that uses Gammapy functionality,
you don't need to know about this or implement it in Gammapy source tree or install it into site-packages.
Just write a ``myscript.py`` file and import the Gammapy functions, classes you need.

TODO: add minimal example using `click`_, then point to the click docs for further information.

Reference/API
=============

.. automodapi:: gammapy.scripts
    :no-inheritance-diagram:
    :include-all-objects:
