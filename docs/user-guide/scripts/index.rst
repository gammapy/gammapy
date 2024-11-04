.. include:: ../../references.txt

.. _CLI:


Command line tools
==================

.. warning::

    The Gammapy command line interface (CLI) described here is experimental
    and only supports a small sub-set of the functionality available via
    the Gammapy Python package.

Currently, Gammapy is first and foremost a Python package. This means that to
use it you have to write a Python script or Jupyter notebook, where you import
the functions and classes needed for a given analysis, and then call them,
passing parameters to configure the analysis.

We have also have a :ref:`analysis` that provides high level Python functions for
the most common needs present in the analysis process.

That said, for some very commonly used and easy to configure analysis tasks we
have implemented a **command line interface (CLI)**. It is automatically
installed together with the Gammapy python package.

Execution
---------

To execute the Gammapy CLI, type the command ``gammapy`` at your terminal shell
(not in Python):

.. code-block:: bash

    gammapy --help

or equivalently, just type this:

.. code-block:: bash

    gammapy

Either way, the command should print some help text to the console and then
exit:

.. code-block:: text

   Usage: gammapy [OPTIONS] COMMAND [ARGS]...

      Gammapy command line interface (CLI).

      Gammapy is a Python package for gamma-ray astronomy.

      Use ``--help`` to see available sub-commands, as well as the available
      arguments and options for each sub-command.

      For further information, see https://gammapy.org/ and
      https://docs.gammapy.org/

      Examples
      --------

      gammapy --help
      gammapy --version
      gammapy info --help
      gammapy info

      Options:
      --log-level [debug|info|warning|error]
                                  Logging verbosity level.
      --ignore-warnings               Ignore warnings?
      --version                       Print version and exit.
      -h, --help                      Show this message and exit.

      Commands:
      analysis  Automation of configuration driven data reduction process.
      check     Run checks for Gammapy
      download  Download datasets and notebooks
      info      Display information about Gammapy

All CLI functionality for Gammapy is implemented as sub-commands of the main
``gammapy`` command. If a command has sub-commands, they are listed in the help
output. E.g. the help output from ``gammapy`` above shows that there is a
sub-command called ``gammapy analysis``. Actually, ``gammapy analysis`` itself isn't a
command that does something, but another command group that is used to group
sub-commands.


So now you know how the Gammapy CLI is structured and how to discover all
available sub-commands, arguments and options.


Running config driven data reduction
------------------------------------

Here's the main usage of the Gammapy CLI for data processing: use the ``gammapy analysis``
command to first create a default configuration file with default values and then
perform a simple automated data reduction process (i.e. fetching observations from
a datastore and producing the reduced datasets.)

.. code-block:: bash

    gammapy analysis --help
    Usage: gammapy analysis [OPTIONS] COMMAND [ARGS]...

    Automation of configuration driven data reduction process.

    Examples
    --------

    gammapy analysis config
    gammapy analysis run
    gammapy analysis config --overwrite
    gammapy analysis config --filename myconfig.yaml
    gammapy analysis run --filename myconfig.yaml

    Options:
    -h, --help  Show this message and exit.

    Commands:
    config  Writes default configuration file.
    run     Performs automated data reduction process.

    gammapy analysis config
    INFO:gammapy.scripts.analysis:Configuration file produced: config.yaml


You can manually edit this produced configuration file and the run the data reduction process:

.. code-block:: bash

    gammapy analysis run

    INFO:gammapy.analysis.config:Setting logging config: {'level': 'INFO', 'filename': None, 'filemode': None, 'format': None, 'datefmt': None}
    INFO:gammapy.analysis.core:Fetching observations.
    INFO:gammapy.analysis.core:Number of selected observations: 4
    INFO:gammapy.analysis.core:Reducing spectrum datasets.
    INFO:gammapy.analysis.core:Processing observation 23592
    INFO:gammapy.analysis.core:Processing observation 23523
    INFO:gammapy.analysis.core:Processing observation 23526
    INFO:gammapy.analysis.core:Processing observation 23559
    Datasets stored in datasets folder.


.. _CLI_write:

Write your own CLI
==================

This section explains how to write your own command line interface (CLI).

We will focus on the command line aspect, and use a very simple example where we
just call `gammapy.stats.CashCountsStatistic.sqrt_ts`.

From the interactive Python or IPython prompt or from a Jupyter notebook you
just import the functionality you need and call it, like this:

   >>> from gammapy.stats import CashCountsStatistic
   >>> CashCountsStatistic(n_on=10, mu_bkg=4.2).sqrt_ts
   np.float64(2.397918129147546)

If you imagine that the actual computation involves many lines of code (and not
just a one-line function call), and that you need to do this computation
frequently, you will probably write a Python script that looks something like
this:

.. testcode::

    # Compute significance for a Poisson count observation
    from gammapy.stats import CashCountsStatistic

    n_observed = 10
    mu_background = 4.2

    s = CashCountsStatistic(n_observed, mu_background).sqrt_ts
    print(f"{s:.4f}")

.. testoutput::

    2.3979

We have introduced variables that hold the parameters for the analysis and put
them before the computation. Let's say this script is in a file called
``significance.py``, then to use it you put the parameters you like and then
execute it via:

.. code-block:: bash

    python significance.py

If you want, you can also put the line ``#!/usr/bin/env python`` at the top of
the script, make it executable via ``chmod +x significance.py`` and then you'll
be able to execute it via ``./significance.py`` if you prefer to execute it like
this. This works on Linux and Mac OS, but not on Windows. It is also possible to
omit the ``.py`` extension from the filename, i.e. to simply call the file
``significance``. Either way has some advantages and disadvantages, it's a
matter of taste. Omitting the ``.py`` is nice because users calling the tool
usually don't care that it's a Python script, and it's shorter. But omitting the
``.py`` also means that some advanced users that open up the file in an editor
have a harder time (because the editor might not recognise it as a Python file
and syntax highlight appropriately), or more importantly that importing
functions of classes from that script from other Python files or Jupyter
notebooks is not easily possible, leading some people to rename it or copy &
paste from it. We're explaining these details, because if you work with
colleagues and share scripts, you'll encounter the ``#!/usr/bin/env python`` and
scripts with and without ``.py`` and will need to know how to work with them.

Writing and using such scripts is perfectly fine and a common way to run science
analyses. However, if you use it very frequently it might become annoying to
have to open up and edit the ``significance.py`` file every time to use it. In
that case, you can change your script into a command line interface that allows
you to set analysis parameters without having to edit the file, like this:

.. code-block:: bash

    python significance.py --help
      Usage: significance.py [OPTIONS] N_OBSERVED MU_BACKGROUND

      Compute significance for a Poisson count observation.

      The significance is the tail probability to observe N_OBSERVED counts or
      more, given a known background level MU_BACKGROUND.

      Options:
      --value [sqrt_ts|p_value]  Square root TS or p_value
      --help                     Show this message and exit.

    python significance.py 10 4.2
    2.39791813

    python significance.py 10 4.2 --value p_value
    0.01648855015875024

In Python, there are several ways to do command line argument parsing and to
create command line interfaces. Of course you're free to do whatever you like,
but if you're not sure what to use to build your own CLIs, we suggest you give
`click`_ a try. Here is how you'd rewrite your ``significance.py`` as a click
CLI:

.. literalinclude:: significance.py

We use `click`_ in Gammapy itself. We also use `click`_ frequently for our own
projects if we choose to add a CLI (no matter if Gammapy is used or not). Putting
the CLI in a file called ``make.py`` makes it easy to go back to a project after
a while and to remember or quickly figure out again how it works (as opposed to
just having a bunch of Python scripts or Jupyter notebooks where it's harder to
remember where to edit parameters and which ones to run in which order). One example
is the `gamma-cat make.py`_.

.. _gamma-cat make.py: https://github.com/gammapy/gamma-cat/blob/master/make.py
.. _gamma-sky.net make.py: https://github.com/gammapy/gamma-sky/blob/master/make.py

If you find that you don't like `click`_, another popular alternative to create
CLIs is `argparse`_ from the Python standard library. To learn argparse, either
read the official documentation, or the `PYMOTW argparse`_ tutorial. For basic
use cases ``argparse`` is similar to ``click``, the main difference being that
``click`` uses decorators (``@command``, ``@argument``, ``@option``) attached to
a callback function to execute, whereas ``argparse`` uses classes and method
calls to create a parser object, and then you have to call ``parse_args``
yourself and also pass the ``args`` to the code or function to execute yourself.
So for basic use cases, but also for more advanced use cases where you define a
CLI with sub-commands, ``argparse`` can be used just as well, it's just a little
harder to learn and use than ``click`` (of course that's a matter of opinion).
Another advantage of choosing Click is that once you've learned it, you'll be
able to quickly read and understand, or even contribute to the Gammapy CLI.

.. _argparse: https://docs.python.org/3/library/argparse.html
.. _PYMOTW argparse: https://pymotw.com/3/argparse/index.html


Troubleshooting
===============

Command not found
-----------------

Usually tools that install Gammapy (e.g. setuptools via ``python setup.py
install`` or ``pip`` or package managers like ``conda``) will put the
``gammapy`` command line tool in a directory that is on your ``PATH``, and if
you type ``gammapy`` the command is found and executed.

However, due to the large number of supported systems (Linux, Mac OS, Windows)
and different ways to install Python packages like Gammapy (e.g. system install,
user install, virtual environments, conda environments) and environments to
launch command line tools like ``gammapy`` (e.g. bash, csh, Windows command
prompt, Jupyter, ...) it is not unheard of that users have trouble running
``gammapy`` after installing it.

This usually looks like this:

.. code-block:: bash

    gammapy
    -bash: gammapy: command not found

If you just installed Gammapy, search the install log for the message
"Installing gammapy script" to see where ``gammapy`` was installed, and check
that this location is on your PATH:

.. code-block:: bash

    echo $PATH

If you don't manage to figure out where the ``gammapy`` command line tool is
installed, you can try calling it like this instead:

.. code-block:: bash

    python -m gammapy

This also has the advantage that it avoids issues where users have multiple
versions of Python and Gammapy installed and accidentally launch one they don't
want because it comes first on their ``PATH``. For the same reason these days
the recommended way to use e.g. ``pip`` is via ``python -m pip``.

If this still doesn't work, check if you are using the right Python and have
Gammapy installed:

.. code-block:: bash

    which python
    python -c 'import gammapy'

To see more information about your shell environment, these commands might be
helpful:

.. code-block:: bash

    python -m site
    python -m gammapy info
    echo $PATH
    conda info -a # if you're using conda

If you're still stuck or have any question, feel free to ask for help with
installation issues on the Gammapy mailing list of Slack any time!

Reference
=========

You may find the auto-generated documentation for all available sub-commands, arguments
and options of the ``gammapy`` command line interface (CLI) in the :ref:`API ref docs <api_CLI>`.
