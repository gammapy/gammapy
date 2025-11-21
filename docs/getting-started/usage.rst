.. include:: ../references.txt

.. _using-gammapy:


Using Gammapy
=============

To use Gammapy you need a basic knowledge of Python, Numpy, Astropy, as well as
matplotlib for plotting. Many standard gamma-ray analyses can be done with a few
lines of configuration and code, so you can get pretty far by copy and pasting
and adapting the working examples from the Gammapy documentation. But
eventually, if you want to script more complex analyses, or inspect analysis
results or intermediate analysis products, you need to acquire a basic to
intermediate Python skill level.

Jupyter notebooks
-----------------

To learn more about Gammapy, and also for interactive data analysis in general,
we recommend you use Jupyter notebooks. Assuming you have followed the steps above to install
Gammapy and activate the conda environment, you can start
`JupyterLab`_ like this:

.. code-block:: bash

    jupyter lab

This should open up JupyterLab app in your web browser, where you can
create new Jupyter notebooks or open up existing ones. If you have downloaded the
tutorials with ``gammapy download tutorials``, you can browse your ``gammapy-tutorials``
folder with Jupyterlab and execute them there.

If you haven't used Jupyter before, try typing ``print("Hello Jupyter")`` in the
first input cell, and use the keyboard shortcut ``SHIFT + ENTER`` to execute it.

Note that one can utilise the ipykernel functionality of Jupyter Notebook to select
a specific pre-installed Gammapy version from your system (see :ref:`quickstart-setup`).

Python
------

Gammapy is a Python package, so you can of course import and use it from Python:

.. code-block:: bash

    python
    Python 3.6.0 | packaged by conda-forge | (default, Feb 10 2017, 07:08:35)
    [GCC 4.2.1 Compatible Apple LLVM 7.3.0 (clang-703.0.31)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from gammapy.stats import CashCountsStatistic
    >>> CashCountsStatistic(n_on=10, mu_bkg=4.2).sqrt_ts
    2.397918129147546

IPython
-------

IPython is nicer to use for interactive analysis:

.. code-block:: bash

    ipython
    Python 3.6.0 | packaged by conda-forge | (default, Feb 10 2017, 07:08:35)
    Type 'copyright', 'credits' or 'license' for more information
    IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

    In [1]: from gammapy.stats import CashCountsStatistic

    In [2]: CashCountsStatistic(n_on=10, mu_bkg=4.2).sqrt_ts
    Out[2]: array([2.39791813])

For example you can use ``?`` to look up **help for any Gammapy function, class or
method** from IPython:

.. code-block:: bash

    In [3]: CashCountsStatistic?

Of course, you can also use the Gammapy online docs if you prefer, clicking in links
(i.e. `gammapy.stats.CashCountsStatistic`) or using *Search the docs* field in the upper left.

As an example, here's how you can create `gammapy.data.DataStore` and
`gammapy.data.EventList` objects and start exploring H.E.S.S. data:

.. testcode::

    from gammapy.data import DataStore
    data_store = DataStore.from_dir('$GAMMAPY_DATA/hess-dl3-dr1/')
    events = data_store.obs(obs_id=23523).events
    print(events)

.. testoutput::

    EventList
    ---------
    <BLANKLINE>
      Instrument       : H.E.S.S. Phase I
      Telescope        : HESS
      Obs. ID          : 23523
    <BLANKLINE>
      Number of events : 7613
      Event rate       : 4.513 1 / s
    <BLANKLINE>
      Time start       : 53343.92234009259
      Time stop        : 53343.94186555556
    <BLANKLINE>
      Min. energy      : 2.44e-01 TeV
      Max. energy      : 1.01e+02 TeV
      Median energy    : 9.53e-01 TeV
    <BLANKLINE>
      Max. offset      : 58.0 deg

Try to make your first plot using the `gammapy.data.EventList.peek` helper method:

.. code-block::

    import matplotlib.pyplot as plt
    events.peek()
    plt.savefig("events.png")


Python scripts
--------------

Another common way to use Gammapy is to write a Python script.
Try it by putting the following code into a file called ``example.py``:

.. testcode::

    """Example Python script using Gammapy"""
    from gammapy.data import DataStore
    data_store = DataStore.from_dir('$GAMMAPY_DATA/hess-dl3-dr1/')
    events = data_store.obs(obs_id=23523).events
    print(events.energy.mean())

.. testoutput::

    4.418007850646973 TeV

You can run it with Python:

.. code-block:: bash

    python example.py
    4.418007850646973 TeV

If you want to continue with interactive data or results analysis after
running some Python code, use IPython like this:

.. code-block:: bash

    ipython -i example.py

For examples how to run Gammapy analyses from Python scripts, see :doc:`/tutorials/scripts/survey_map`.
