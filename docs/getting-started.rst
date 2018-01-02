.. include:: references.txt

.. _getting-started:

Getting Started
===============

If you'd like to get started using Gammapy, you've come to the right place!

Reading through this page will just take you a few minutes.

But we hope that you'll get curious and start executing the examples yourself,
using Gammapy to analyse (simulated) H.E.S.S. and real Fermi-LAT data.

If you're new to Python for gamma-ray astronomy and would like to learn the basics, we recommend
you go to the `Scipy Lecture Notes`_ or the `Practical Python for Astronomers Tutorial`_.

Gammapy as a Python package and set of science tools
----------------------------------------------------

Gammapy is a Python package, consisting of functions and classes, that you can use
as a flexible and extensible toolbox to implement and execute exactly the analysis you want.

On top of that, Gammapy provides some command line tools (sometimes driven by a config file),
and in the future we plan on adding web apps with a graphical user interface.
To use those no Python programming skills are required, you'll just have to specify which
data to analyse, with which method and parameters.


Getting set up
--------------

First, make sure you have Gammapy installed (see :ref:`install`).

You can use this command to make sure the Python package is available::

   $ python -c 'import gammapy'

To check if the Gammapy command line tool has been installed and are available on your PATH, use this command::

    $ gammapy --version

The Gammapy tutorials use some example datasets that are stored in the ``gammapy-extra`` repository on Github.
So please go follow the instructions at :ref:`gammapyextra` to fetch those, then come back here.

To check if ``gammapy-extra`` is available and the ``GAMMAPY_EXTRA`` shell environment variable set, use this command::

    $ echo $GAMMAPY_EXTRA
    $ ls $GAMMAPY_EXTRA/logo/gammapy_banner.png

Need help?
----------

If you have any questions or issues with installation, setup or Gammapy usage,
lease use the `Gammapy mailing list`_!

Gammapy is a very young project, we know there are many missing features and issues.
Please have some patience, and let us know what you want to do, so that we can set priorities.


Using Gammapy as a Python package
---------------------------------

Here's a few very simple examples how to use Gammapy as a Python package.

What's the statistical significance when 10 events have been observed with a known background level of 4.2
according to [LiMa1983]_?

Getting the answer from Gammapy is easy. You import and call the `gammapy.stats.significance` function:

.. code-block:: python

   >>> from gammapy.stats import significance
   >>> significance(n_observed=10, mu_background=4.2, method='lima')
   2.3979181291475453

As another example, here's how you can create `gammapy.data.DataStore` and `gammapy.data.EventList`
objects and start exploring some properties of the (simulated) H.E.S.S. event data:

.. code-block:: python

   >>> from gammapy.data import DataStore
   >>> data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
   >>> events = data_store.obs(obs_id=23523).events
   >>> len(events)
   1527
   >>> events.energy.mean()
   <Quantity 3.585395097732544 TeV>


How do you find something in Gammapy?

Often using the full-text search field is the quickest and simplest way.
As you get to know the package, you'll learn the names of the different sub-packages that are available,
like `gammapy.stats` or `gammapy.data`, and what functionality they contain.

Another good way is tutorials, IPython and Jupyter notebooks ...

Using Gammapy from the Jupyter notebooks
----------------------------------------

In the last section you've seen how to use Gammapy as a Python package.
To become good at using it, you have to learn the Gammapy API (application programming interface).
One way to do this is to read documentation. A more interactive (and arguably more fun)
way is to play with Gammapy code and gamma-ray data in Jupyter notebooks.

Jupyter notebooks are documents that combine code input and text and graphical output,
and are wonderful tools to learn and explore (both programming and the data),
and finally to share results with your colleagues.

So now is a good time to have a look here: :ref:`tutorials`.
Try executing the cells locally on your machine as you read through the text and code.


Using Gammapy via command line tools
------------------------------------

The ``gammapy`` command line tool lets you execute some very common analysis tasks
directly from the command line. Try this::

    $ gammapy --help
    $ gammapy --version

Further information about the ``gammapy`` command line interface is here: :ref:`scripts`

That page also includes information how the ``gammapy`` command line tool works
and what to do if it doesn't work (likely you have to add the ``bin`` directory
where Gammapy is installed to your ``PATH`` shell environment variable). It even
has a section with a tutorial how to write your own command line tools if this is
something you want.

What next?
----------

If you'd like to continue with tutorials to learn Gammapy, go here: :ref:`tutorials`.

To learn about some specific functionality that could be useful for your work,
start browsing the "Getting Started" section of Gammapy sub-package that
might be of interest to you (e.g. `gammapy.data`, `gammapy.catalog`, `gammapy.spectrum`, ...).
