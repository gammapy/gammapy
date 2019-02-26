.. include:: references.txt

.. _getting-started:

Getting Started
===============

The best way to learn about Gammapy is to read and play with the examples in the
Gammapy :ref:`tutorials`.

This section explains the steps to get set up for the Gammapy tutorials on your
machine:

1. Install Anaconda and the Gammapy environment
2. Download tutorial notebooks and example datasets
3. Check your setup and Gammapy environment
4. Use Gammapy with Python, IPython or Jupyter

If you have used conda, Python, IPython and Jupyter before, you can just skim
this page, and quickly copy & paste the commands to get set up.

Help!?
------

If you have any questions or issues, please ask for help on the Gammapy Slack,
mailing list or on Github (whatever is easiest for you, see `Gammapy contact`_)

Install
-------

To install Gammapy we recommend that you install the Anaconda distribution from
https://www.anaconda.com/download/. The following instructions and commands below
need to have Anaconda installed.

Anaconda it's free, works on Linux, MacOS and Windows. It installs in your home
directory, no system privileges needed, and you can just delete it without
problems if you don't need or want it any more.

Besides the software included in the Anaconda distribution, it gives you the
`conda`_ tool, which is a package and environment manager. We will use it to
create a dedicated environment for the latest stable Gammapy version and a known
good set of Gammapy dependencies (e.g. Python, Numpy and Astropy).

Once Anaconda has been installed, use the following commands to install and activate
the ``gammapy-0.10`` conda environment:

.. code-block:: bash

    curl -O https://gammapy.org/download/install/gammapy-0.10-environment.yml
    conda env create -f gammapy-0.10-environment.yml
    conda activate gammapy-0.10

On Windows, comment out the lines for ``sherpa`` and ``healpy`` in the
environment YAML file. These are optional dependencies that we are currently
using that aren't supported on Windows. You will be able to do most analyses
with Gammapy, but you will not be able to work with HEALPix data or fit
with Sherpa (the default fitting backend currently is Minuit).

Congratulations! You are all set to start using Gammapy!

.. note::

    Every time you open a new terminal window, you will have to activate
    the Gammapy conda environment again before you can use it.

.. code-block:: bash

    conda activate gammapy-0.10

Download tutorials
------------------

You can now proceed to download the Gammapy tutorial notebooks and the example
datasets used there (at the moment from CTA, H.E.S.S. and Fermi-LAT). The total
size to download is about 100 MB.

.. code-block:: bash

    gammapy download tutorials
    cd gammapy-tutorials
    export GAMMAPY_DATA=$PWD/datasets

You might want to put the definition of the ``$GAMMAPY_DATA`` environment
variable in your shell profile setup file that is executed when you open a new
terminal (for example ``$HOME/.bash_profile``).

If you are not using the ``bash`` shell, handling of shell environment variables
might be different, e.g. in some shells the command to use is ``set`` or something
else instead of ``export``, and also the profile setup file will be different.

On Windows, you should set the ``GAMMAPY_DATA`` environment variable in the
"Environment Variables" settings dialog, as explained e.g.
`here <https://docs.python.org/3/using/windows.html#excursus-setting-environment-variables>`__

The datasets are curated and stable, the notebooks are still under development
just like Gammapy itself, and thus stored in a sub-folder that contains the
Gammapy version number.

The ``gammapy download`` command and versioning of the notebooks is new. If
there are issues, note that you can just delete the folder any time using ``rm
-r gammapy-tutorials`` and start over.

Also note that it's of course possible to download just the notebooks, just
the datasets, a single notebook or dataset, or a specific tutorial with the
datasets it uses. See the help:

.. code-block:: bash

    gammapy download --help

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


Use Gammapy
-----------

Python
++++++

Gammapy is a Python package, so you can of course import and use it from Python:

.. code-block:: bash

    $ python
    Python 3.6.0 | packaged by conda-forge | (default, Feb 10 2017, 07:08:35)
    [GCC 4.2.1 Compatible Apple LLVM 7.3.0 (clang-703.0.31)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from gammapy.stats import significance
    >>> significance(n_on=10, mu_bkg=4.2, method='lima')
    array([2.39791813])

IPython
+++++++

IPython is nicer to use for interactive analysis:

.. code-block:: bash

    $ ipython
    Python 3.6.0 | packaged by conda-forge | (default, Feb 10 2017, 07:08:35)
    Type 'copyright', 'credits' or 'license' for more information
    IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

    In [1]: from gammapy.stats import significance

    In [2]: significance(n_on=10, mu_bkg=4.2, method='lima')
    Out[2]: array([2.39791813])

For example you can use ``?`` to look up **help for any Gammapy function, class or
method** from IPython:

.. code-block:: bash

    In [3]: significance?

Of course, you can also use the Gammapy online docs if you prefer, clicking in links
(i.e. `gammapy.stats.significance`) or using *search docs* field in the upper left.

As an example, here's how you can create `gammapy.data.DataStore` and
`gammapy.data.EventList` objects and start exploring H.E.S.S. data:

.. code-block:: python

    >>> from gammapy.data import DataStore
    >>> data_store = DataStore.from_dir('$GAMMAPY_DATA/hess-dl3-dr1/')
    >>> events = data_store.obs(obs_id=23523).events
    >>> print(events)
    EventList info:
    - Number of events: 7613
    - Median energy: 0.953 TeV
    - OBS_ID = 23523
    >>> events.energy.mean()
    <Quantity 4.418008 TeV>

Try to make your first plot using the `peek` helper method:

.. code-block:: python

    >>> events.peek()
    >>> plt.savefig("events.png")

Python script
+++++++++++++

Another common way to use Gammapy is to write a Python script.
Try it and put the following code into a file called ``example.py``:

.. code-block:: python

    """Example Python script using Gammapy"""
    from gammapy.data import DataStore
    data_store = DataStore.from_dir('$GAMMAPY_DATA/hess-dl3-dr1/')
    events = data_store.obs(obs_id=23523).events
    print(events.energy.mean())

You can run it with Python:

.. code-block:: bash

    $ python example.py
    4.418007850646973 TeV

If you want to continue with interactive data or results analysis after
running some Python code, use IPython like this:

.. code-block:: bash

    $ ipython -i example.py

Command line
++++++++++++

As you have already seen, installing Gammapy gives you a ``gammapy`` command line
tool with subcommands such as ``gammapy info`` or ``gammapy download``.

We plan to add a high-level interface to Gammapy soon that will let you run
Gammapy analyses via the command line interface, probably driven by a config
file. This is not available yet, for now you have to use Gammapy as a Python
package.

Jupyter notebooks
+++++++++++++++++

To learn more about Gammapy, and also for interactive data analysis in general,
we recommend you use Jupyter notebooks. Assuming you have followed the steps above to install Gammapy and activate the conda environment, you can start
`JupyterLab`_ like this:

.. code-block:: bash

    $ jupyter lab

This should open up JupyterLab app in your web browser, where you can
create new Jupyter notebooks or open up existing ones. If you have downloaded the
tutorials with `gammapy download tutorials`, you can browse your `gammapy-tutorials`
folder with Jupyterlab and execute them there.

If you haven't used Jupyter before, try typing ``print("Hello Jupyter")`` in the
first input cell, and use the keyboard shortcut ``SHIFT + ENTER`` to execute it.

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
