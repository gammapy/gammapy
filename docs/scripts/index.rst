.. include:: ../references.txt

.. _scripts:

****************************
scripts - Command line tools
****************************

.. currentmodule:: gammapy.scripts

.. warning::

    The Gammapy command line interface (CLI) described here is experimental
    and only supports a small sub-set of the functionality available via
    the Gammapy Python package. We have added a few sections at the bottom
    of this page to explain the current :ref:`scripts_implementation`,
    :ref:`scripts_limitations` and :ref:`scripts_plan`. And since we don't
    offer much here yet, at least we describe how you can :ref:`scripts_user_cli`.

.. _scripts_intro:

Introduction
============

Currently, Gammapy is first and foremost a Python package. This means that to
use it you have to write a Python script or Jupyter notebook, where you import
the functions and classes needed for a given analysis, and then call them,
passing parameters to configure the analysis.

That said, for some very commonly used and easy to configure analysis tasks we
have implemented a **command line interface (CLI)**. It is automatically
installed together with the Gammapy python package.

Execute
-------

To execute the Gammapy CLI, type the command ``gammapy`` at your terminal shell
(not in Python)::

    $ gammapy --help

or equivalently, just type this::

    $ gammapy

Either way, the command should print some help text to the console and then
exit:

.. code-block:: text

    $ gammapy

      Gammapy command line interface.

      Gammapy is a Python package for gamma-ray astronomy.

      For further information, see https://gammapy.org/

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

All CLI functionality for Gammapy is implemented as sub-commands of the main
``gammapy`` command. If a command has sub-commands, they are listed in the help
output. E.g. the help output from ``gammapy`` above shows that there is a
sub-command called ``gammapy image``. Actually, ``gammapy image`` itself isn't a
command that does something, but another command group that is used to group
sub-commands:

.. code-block:: text

    $ gammapy image
    Usage: gammapy image [OPTIONS] COMMAND [ARGS]...

      Analysis - 2D images

    Options:
      -h, --help  Show this message and exit.

    Commands:
      bin  Bin events into an image.
      fit  Fit morphology model to image using Sherpa.
      ts   Compute TS image.

Finally, ``gammapy image bin`` is a proper sub-sub-command that does something,
it doesn't have any sub-commands itself, just arguments and options. If you call
it without passing the required arguments, you will get an error:

.. code-block:: text

    $ gammapy image bin
    Usage: gammapy image bin [OPTIONS] EVENT_FILE REFERENCE_FILE OUT_FILE

    Error: Missing argument "event_file".

Use ``--help`` to see the help text and available options:

.. code-block:: text

    $ gammapy image bin --help
    Usage: gammapy image bin [OPTIONS] EVENT_FILE REFERENCE_FILE OUT_FILE

      Bin events into an image.

      You have to give the event, reference and out FITS filename.

    Options:
      --overwrite  Overwrite existing files?
      -h, --help   Show this message and exit.

So now you know how the Gammapy CLI is structured and how to discover all
available sub-commands, arguments and options.

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

This usually looks like this::

    $ gammapy
    -bash: gammapy: command not found

If you just installed Gammapy, search the install log for the message
"Installing gammapy script" to see where ``gammapy`` was installed, and check
that this location is on your PATH::

    echo $PATH

If you don't manage to figure out where the ``gammapy`` command line tool is
installed, you can try calling it like this instead::

    $ python -m gammapy

This also has the advantage that it avoids issues where users have multiple
versions of Python and Gammapy installed and accidentally launch one they don't
want because it comes first on their ``PATH``. For the same reason these days
the recommended way to use e.g. ``pip`` is via ``python -m pip``.

If this still doesn't work, check if you are using the right Python and have
Gammapy installed::

    $ which python
    $ python -c 'import gammapy'

To see more information about your shell environment, these commands might be
helpful::

    $ python -m site
    $ python -m gammapy info
    $ echo $PATH
    $ conda info -a # if you're using conda

If you're still stuck or have any question, feel free to ask for help with
installation issues on the Gammapy mailing list of Slack any time!

Example
-------

Here's one example what you can do with the Gammapy CLI: use the ``gammapy image
bin`` command to create a counts image from a Fermi-LAT event list as well as a
FITS image that serves as a reference geometry (projection, center, binning) for
the counts image we're creating.

.. code-block:: text

    $ gammapy image bin --help
    Usage: gammapy image bin [OPTIONS] EVENT_FILE REFERENCE_FILE OUT_FILE

      Bin events into an image.

      You have to give the event, reference and out FITS filename.

    Options:
      --overwrite  Overwrite existing files?
      -h, --help   Show this message and exit.

    $ gammapy image bin \
        $GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz \
        $GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz \
        out.fits
    INFO:gammapy.scripts.image_bin:Executing cli_image_bin
    INFO:gammapy.scripts.image_bin:Reading /home/gammapy-data/datasets/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz
    INFO:gammapy.scripts.image_bin:Reading /home/gammapy-data/datasets/fermi-3fhl-gc/fermi-3fhl-gc-counts.fits.gz
    INFO:gammapy.scripts.image_bin:Writing out.fits

If you have the FTOOLS_ installed or other tools that can work with the files
that Gammapy supports, you can of course use them together::

    $ ftlist $GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz H

            Name               Type       Dimensions
            ----               ----       ----------
    HDU 1   Primary Array      Null Array
    HDU 2   EVENTS             BinTable    23 cols x 32843 rows
    HDU 3   GTI                BinTable     2 cols x 39042 rows


    $ ftlist out.fits H

            Name               Type       Dimensions
            ----               ----       ----------
    HDU 1   Primary Array      Image      Real4(400x200)

    $ ds9 out.fits

.. _scripts_ref:

Reference
=========

Here is auto-generated documentation for all available sub-commands, arguments
and options of the ``gammapy`` command line interface (CLI).

It's not very readable at the moment. With the current formatting it's a bit
hard to tell where documentation for a new sub-command starts and what level of
subcommand one is looking at for a given heading. Maybe change to one page per
sub-command?

.. click:: gammapy.scripts.main:cli
   :prog: gammapy
   :show-nested:

.. _scripts_implementation:

Implementation
==============

Currently, the command line interface (CLI) of Gammapy is implemented using
`click`_ to define sub-commands, arguments and options, as well as calling the
right function that implements a given sub-command.

We have chosen to implement all functionality via a single command line tool
called ``gammapy``, with each task as a subcommand. The ``gammapy`` command line
tool uses the `setuptools console_scripts entry point`_ method to automatically
create command line tools when Gammapy is installed.

This means that to be able to use the tools you have to install Gammapy.
Although, from the source folder you can still execute it without installing via

.. code-block:: bash

    $ python -m gammapy

which executes ``gammapy/__main__.py`` as a script as explained `here
<https://docs.python.org/3/library/__main__.html>`__.

Another way to install the ``gammapy`` command line tool once, but to have it
point at the Gammapy git source folder while you're hacking on Gammapy is to use

.. code-block:: bash

    $ python -m pip install --editable .

Either ``gammapy`` or ``python -m gammapy`` call the
``gammapy.scripts.main.cli`` function, which is a ``click.Group`` object. If you
want you can also import and execute it yourself::

    >>> from gammapy.scripts.main import cli
    >>> type(cli)
    click.core.Group
    >>> cli()
    >>> cli(['--version'])
    >>> cli(['image', 'bin', '--help'])

This is what we do to test the CLI, we import ``gammapy.scripts.main.cli`` and
run it via ``gammapy.utils.testing.run_cli`` and check the return code and
sometimes console output or generated files. Note that this means that all tests
run in a single Python process, we don't "shell out" and create a subprocess
that calls ``gammapy`` from a sub-shell.

This is also how the auto-generated CLI documentation in the :ref:`scripts_ref`
section above was generated: we use the `sphinx-click`_ Sphinx extension that
imports and inspects ``gammapy.scripts.main.cli`` to find out about all the
available sub-commands and help text / arguments / options. The ``click.Group``
object exposes all information as attributes and methods. To just give one
example::

    >>> from gammapy.scripts.main import cli
    >>> cli.commands
    {'check': <click.core.Group at 0x112107048>,
     'image': <click.core.Group at 0x112102e10>,
     'info': <click.core.Command at 0x1120bb278>}

If you're new to Python command line tools or Click, probably setuptools entry
points and click groups seem very complex. And they are, compared to the rest of
Gammapy which is just ``def`` and ``class`` statements to make functions and
classes, i.e. "normal" Python code. Just know that to use and even work on the
Gammapy CLI you don't have to understand how it works under the hood, but if you
want to, it's actually not that complex.

A good path to learn is to start by reading `gammapy/__main__.py`_ and
`gammapy/scripts/main.py`_ and then to look at an example of how a sub-command
in Gammapy is implemented and tested,
e.g. `gammapy/scripts/image_bin.py`_ and `gammapy/scripts/tests/test_image_bin.py`_.

Note how sub-commands are ``click.Command`` objects::

    >>> from gammapy.scripts.image_bin import cli_image_bin
    >>> type(cli_image_bin)
    click.core.Command

that are independent and how the main ``cli`` is created via ``cli.add_command``
calls in `gammapy/scripts/main.py`_.

Now you have a basic understanding how things work and should be able to work on
Gammapy CLI (e.g. add more functionality to the CLI interface). If you're
curious to learn how it works in detail, we suggest you read the `click`_ docs
and play with the Gammapy ``cli`` object in IPython like we did above when
looking at ``cli.commands``, or read and play with the standalone example in the
:ref:`scripts_user_cli` section below.

.. _gammapy/__main__.py: https://github.com/gammapy/gammapy/blob/master/gammapy/__main__.py
.. _gammapy/scripts/main.py: https://github.com/gammapy/gammapy/blob/master/gammapy/scripts/main.py
.. _gammapy/scripts/image_bin.py: https://github.com/gammapy/gammapy/blob/master/gammapy/scripts/image_bin.py
.. _gammapy/scripts/tests/test_image_bin.py: https://github.com/gammapy/gammapy/blob/master/gammapy/scripts/tests/test_image_bin.py

.. _scripts_limitations:

Limitations
===========

The current `click`_-based Gammapy CLI is pretty nice, it is very simple to add
commands and also documenting and testing them is pretty nice.

However, the current implementation has some issues and limitations. We describe
them in this section, and then in the next one discuss more generally the plan
and options for the Gammapy high-level (CLI or non-CLI) interface.

* There is no support for in-memory tool chain analysis pipelines.
  I mean something lik e.g. ``gammapy bin`` followed by ``gammapy fit``
  without writing intermediate files and starting the two commands as separate processes.
* There is no support for configuring or writing provenance information
  for commands or command pipelines (i.e. store which commands were executed with
  which arguments in input config or "workflow" files as well as output result files).
* More generally, we have to see if the separation of Gammapy as a library
  of "normal" Python functions and classes that can't be driven by config
  files or the command line, and then separate functions that represent the CLI
  is what we want. It means that we have two different ways to use Gammapy
  in very different ways, and there is duplication and not a nice transition
  and re-use between the two ways.

More technical issues that can certainly be fixed if we want to stick with the
current click-based CLI:

* ``gammapy`` always imports all code from all sub-commands, which drags in
  large fraction of ``astropy`` and ``gammapy`` whether it is used or not.
  This means that ``gammapy --help`` takes a few seconds, whereas other commands
  like ``git --help`` just take a very small fraction of a second.
  This can be improved either by optimising import times throughout Astropy
  and Gammapy in general, or by lazy-loading the subcommands or by delaying
  imports into the callbacks from the subcommands.
* The CLI documentation isn't nice yet (see the :ref:`scripts_ref` section above).
  The `sphinx-click`_ package that we use isn't very well-developed or configurable.
  However, it's a single Python file that we could just copy into Gammapy and
  modify and extend to generate documentation in exactly the way we like.
  E.g. we probably would want to have help text including examples and links to
  other parts for the Gammapy API and CLI docs that appear nicely on the console
  as well on in the HTML docs.

.. _scripts_plan:

Plan
====

There is no concrete plan yet concerning the high-level user interface for
Gammapy. Feedback from users and developers on the mailing list is highly
welcome! What do you want?

Some options we are considering to build the high-level end-user interface for
Gammapy:

1. Collection of command line tools. Examples: FTOOLs_, `Fermi ScienceTools`_, `ctools`_
2. Config-file driven analysis. Examples: `FermiPy`_, or the H.E.S.S.-internal HAP
3. No special config- or CLI interface, just normal Python functions and classes.
   Examples: Sherpa_ and most Python package like e.g. `Astropy`_ or `scikit-learn`_.
4. Something more fancy that supports tool configuration and running from Python,
   config files or a CLI using a single implementation. Examples: `ctapipe`_, `fact-tools`_,
   `python-fire`_

Options 1 and 2 are nice and simple, and they are user-friendly interfaces, and
they would allow us to have a stable high-level interface while being able to
continue to improve the Gammapy package without breaking user scripts over the
coming years.

Their drawback is that they create a second way to use Gammapy, with some users
learning and using the more flexible and powerful Python package, and some the
simpler to use, but less flexible high-level interface. And note that eventually
this second interface will grow into a config file with 100 options (that's what
we have in HAP) or into 10s of CLI tools with in total again 100s of options
(see the ctools). However, the Fermi Science tools CLI and Fermipy config
interface are examples where the interface size remains at a reasonable level
(for users to learn and for developers to implement and maintain) while still
exposing everything that most users need.

Options 3 and 4 are similar, in either case the analysis functionality that is
available in Gammapy would be written once. Option 3 is what we have now in the
Gammapy Python package. Changing to option 4 would mean adding some boilerplate
code everywhere (e.g. Python decorators or sub-classing from "tool" base classes
like what ctapipe is developing) or relying on the dynamic and inspection
features of the Python language offers (see e.g. python-fire), to make it
possible to configure and drive analyses not just via Python code, but via some
configuration coming either from configuration files (YAML or XML) or command
line options.

So what should we do for Gammapy?

I would suggest we continue with the prototyping of the CLI interface as well as
of a config-file based interface (`gammapy.scripts.SpectrumAnalysisIACT` is a
starting point). Pull requests that improve and extend what we have are welcome
any time!

In parallel, we continue to learn and evaluate solutions others have developed.
In `python-cli-examples`_ I have started an exploration and evaluation of Python
CLI packages, namely `click`_, but also `cliff`_ and `traitlets`_, and I might
take a closer look also at `cement`_ and `python-fire`_ there. We should also
look in detail at what other projects like LSST, JWST, Fermi, ctapipe, and
others do concerning configuration and interface of their science tools. Later
in 2018 we will need a comprehensive proposal for code organisation and
high-level interface for Gammapy. Suggestions or even contributions are welcome
any time!

.. _fact-tools: https://pos.sissa.it/236/865/
.. _cement: http://builtoncement.com/
.. _python-fire: https://github.com/google/python-fire
.. _python-cli-examples: https://github.com/cdeil/python-cli-examples/
.. _cliff: https://docs.openstack.org/cliff/latest/
.. _traitlets: http://traitlets.readthedocs.io/


.. _scripts_user_cli:

Write your own CLI
==================

This section explains how to write your own command line interface (CLI).

We will focus on the command line aspect, and use a very simple example where we
just call `gammapy.stats.significance`.

From the interactive Python or IPython prompt or from a Jupyter notebook you
just import the functionality you need and call it, like this:

   >>> from gammapy.stats import significance
   >>> significance(n_observed=10, mu_background=4.2, method='lima')
   2.3979181291475453

If you imagine that the actual computation involves many lines of code (and not
just a one-line function call), and that you need to do this computation
frequently, you will probably write a Python script that looks something like
this:

.. code-block:: python

    # Compute significance for a Poisson count observation
    from gammapy.stats import significance

    n_observed = 10
    mu_background = 4.2
    method = 'lima'

    s = significance(n_observed, mu_background, method)
    print(s)

We have introduced variables that hold the parameters for the analysis and put
them before the computation. Let's say this script is in a file called
``significance.py``, then to use it you put the parameters you like and then
execute it via::

    $ python significance.py

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
you to set analysis parameters without having to edit the file, like this::

    $ python significance.py --help
    Usage: significance.py [OPTIONS] N_OBSERVED MU_BACKGROUND

      Compute significance for a Poisson count observation.

      The significance is the tail probability to observe N_OBSERVED counts or
      more, given a known background level MU_BACKGROUND.

    Options:
      --method [lima|simple]  Significance computation method
      --help                  Show this message and exit.

    $ python significance.py 10 4.2
    2.39791813

    $ python significance.py 10 4.2 --method simple
    2.83011021

In Python, there are several ways to do command line argument parsing and to
create command line interfaces. Of course you're free to do whatever you like,
but if you're not sure what to use to build your own CLIs, we suggest you give
`click`_ a try. Here is how you'd rewrite your ``significance.py`` as a click
CLI:

.. literalinclude:: significance.py

As mentioned above in the :ref:`scripts_implementation`, we use `click`_ in
Gammapy itself. We also use `click`_ frequently for our own projects if we
choose to add a CLI (no matter if Gammapy is used or not). Putting the CLI in a
file called ``make.py`` makes it easy to go back to a project after a while and
to remember or quickly figure out again how it works (as opposed to just having
a bunch of Python scripts or Jupyter notebooks where it's harder to remember
where to edit parameters and which ones to run in which order). One example is
the `gamma-cat make.py`_.

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

Reference/API
=============

Besides the CLI interface, the `gammapy.scripts` package currently contains a
bunch of things that will probably all be removed or rewritten and integrated in
other sub-packages of Gammapy, leaving ``scripts`` just as the high-level
command line script interface for Gammapy.

.. automodapi:: gammapy.scripts
    :no-inheritance-diagram:
    :include-all-objects:
