.. include:: ../references.txt

.. _dev_howto:

***************
Developer HOWTO
***************

This page is a collection of notes for Gammapy contributors and maintainers,
in the form of short "How to" or "Q & A" entries.


.. _make_clean:

How to clean up old files
-------------------------

TODO: Gammapy now has a Makefile ... this section should be expanded to a page about setup.py and make.

Many projects have a ``Makefile`` to build and install the software and do all kinds of other tasks.
In Astropy and Gammapy and most Python projects, there is no ``Makefile``, but the ``setup.py`` file
and you're supposed to type ``python setup.py <cmd>`` and use ``--help`` and ``--help-commands`` to
see all the available commands and options.

There's one common task, cleaning up old generated files, that's not done via ``setup.py``.
The equivalent of ``make clean`` is:

.. code-block:: bash

    $ rm -r build docs/_build docs/api htmlcov docs/notebooks docs/_static/notebooks

These folders only contain generated files and are always safe to delete!
Most of the time you don't have to delete them, but if you e.g. remove or rename files or functions / classes,
then you should, because otherwise the old files will still be around and you might get confusing results,
such as Sphinx warnings or import errors or code that works locally because it uses old things, but fails
on travis-ci or for other developers.

* The ``build`` folder is where ``python setup.py build`` or ``python setup.py install`` generate files.
* The ``docs/api`` folder is where ``python setup.py build_docs`` generates [RST]_ files from the docstrings
  (temporary files part of the HTML documentation generation).
* The  ``docs/_build`` folder is where ``python setup.py build_docs`` generates the HTML and other Sphinx
  documentation output files.
* The ``htmlcov`` folder is where ``python setup.py test --coverage`` generates the HTML coverage report.
* The ``docs/notebooks`` and ``docs/_static/notebooks`` folders are where *fixed* and *live* versions of Jupyter notebooks files are stored.

If you use ``python setup.py build_ext --inplace``, then files are generated in the ``gammapy`` source folder.
Usually that's not a problem, but if you want to clean up those generated files, you can use
`git clean <http://git-scm.com/docs/git-clean>`__:

.. code-block:: bash

    $ git status
    # The following command will remove all untracked files!
    # If you have written code that is not committed yet in a new file it will be gone!
    # So use with caution!
    $ git clean -fdx

At least for now we prefer not to add a ``Makefile`` to Gammapy, because that splits the developers into
those that use ``setup.py`` and those that use ``make``, which can grow into an overall **more** complicated
system where some things are possible only with ``setup.py`` and others only with ``make``.

.. _dev_import:

Where should I import from?
---------------------------

You should import from the "end-user namespaces", not the "implementation module".

.. code-block:: python

   from gammapy.data import EventList  # good
   from gammapy.data.event_list import EventList # bad

   from gammapy.stats import cash  # good
   from gammapy.stats.fit_statistics import cash  # bad

The end-user namespace is the location that is shown in the API docs, i.e. you can
use the Sphinx full-text search to quickly find it.

To make code maintenance easier, the implementation of the functions and classes is
spread across multiple modules (``.py`` files), but the user shouldn't care about their
names, that would be too much to remember.

The only reason to import from a module directly is if you need to access a private
function, class or variable (something that is not listed in ``__all__`` and thus not
imported into the end-user namespace.

Note that this means that in the definition of an "end-user namespace", e.g. in the
``gammapy/data/__init__.py`` file, the imports have to be sorted in a way such that
modules in ``gammapy/data`` are loaded when imported from other modules in that sub-package.

.. _dev-result_object:

Functions returning several values
----------------------------------

Functions that return more than a single value shouldn't return a list
or dictionary of values but rather a Python Bunch result object. A Bunch
is similar to a dict, except that it allows attribute access to the result
values. The approach is the same as e.g. the use of `~scipy.optimize.OptimizeResult`.
An example of how Bunches are used in gammapy is given by the `~gammapy.image.SkyImageList`
class.

.. _dev-python2and3:

Python 2 and 3 support
----------------------

We support Python 2.7 and 3.4 or later using a single code base.
This is the strategy adopted by most scientific Python projects and a good starting point to learn about it is
`here <http://python3porting.com/noconv.html>`__ and
`here <http://docs.astropy.org/en/latest/development/codeguide.html#writing-portable-code-for-python-2-and-3>`__.

For developers, it would have been nice to only support Python 3 in Gammapy.
But the CIAO and Fermi Science tools software are distributed with Python 2.7
and probably never will be updated to Python 3.
Plus many potential users will likely keep running on Python 2.7 for many years
(see e.g. `this survey <http://ipython.org/usersurvey2013.html#python-versions>`__).

The decision to drop Python 2.6 and 3.2 support was made in August 2014 just before the Gammapy 0.1 release,
based on a few scientific Python user surveys on the web that show that only a small minority are still
using such an old version, so that it's not worth the developer and maintainer effort to test
these old versions and to find workarounds for their missing features or bugs.

Python 3.3 support was dropped in August 2015 because conda packages for some of the affiliated packages
weren't available for testing on travis-ci.

.. _dev-skip_tests:

Skip unit tests for some Astropy versions
-----------------------------------------

.. code-block:: python

   import astropy
   import pytest

   ASTROPY_VERSION = (astropy.version.major, astropy.version.minor)
   @pytest.mark.xfail(ASTROPY_VERSION < (0, 4), reason="Astropy API change")
   def test_something():
      ...

Fix non-Unix line endings
-------------------------

In the past we had non-Unix (i.e. Mac or Windows) line endings in some files.
This can be painful, e.g. git diff and autopep8 behave strangely.
Here's to commands to check for and fix this (see `here <http://stackoverflow.com/a/22521008/498873>`__):

.. code-block:: bash

    $ git clean -fdx
    $ find . -type f -print0 | xargs -0 -n 1 -P 4 dos2unix -c mac
    $ find . -type f -print0 | xargs -0 -n 1 -P 4 dos2unix -c ascii
    $ git status
    $ cd astropy_helpers && git checkout -- . && cd ..

.. _dev-check_html_links:

Check HTML links
----------------

To check for broken external links from the Sphinx documentation:

.. code-block:: bash

   $ python setup.py install
   $ cd docs; make linkcheck

Other codes
-----------

These projects are on Github, which is great because
it has full-text search and git history view:

* https://github.com/gammapy/gammapy
* https://github.com/gammapy/gammapy-extra
* https://github.com/astropy/astropy
* https://github.com/astropy/photutils
* https://github.com/gammalib/gammalib
* https://github.com/ctools/ctools
* https://github.com/sherpa/sherpa
* https://github.com/zblz/naima
* https://github.com/woodmd/gammatools
* https://github.com/fermiPy/fermipy
* https://github.com/kialio/VHEObserverTools
* https://github.com/taldcroft/xrayevents

These are unofficial, unmaintained copies on open codes on Github:

* https://github.com/Xarthisius/yt-drone
* https://github.com/cdeil/Fermi-ScienceTools-mirror

Actually at this point we welcome experimentation, so you can use cool new technologies
to implement some functionality in Gammapy if you like, e.g.

* `Numba <http://numba.pydata.org/>`__
* `Bokeh <http://bokeh.pydata.org/en/latest/>`__
* `Blaze <http://blaze.pydata.org/en/latest/>`__


What checks and conversions should I do for inputs?
---------------------------------------------------

In Gammapy we assume that
`"we're all consenting adults" <https://mail.python.org/pipermail/tutor/2003-October/025932.html>`_,
which means that when you write a function you should write it like this:

.. code-block:: python

    def do_something(data, option):
        """Do something.

        Parameters
        ----------
        data : `numpy.ndarray`
            Data
        option : {'this', 'that'}
            Option
        """
        if option == 'this':
            out = 3 * data
        elif option == 'that':
            out = data ** 5
        else:
            ValueError('Invalid option: {}'.format(option))

        return out

* **Don't always add `isinstance` checks for everything** ... assume the caller passes valid inputs,
  ... in the example above this is not needed::

        assert isinstance(option, str)

* **Don't always add `numpy.asanyarray` calls for all array-like inputs** ... the caller can do this if
  it's really needed ... in the example above document ``data`` as type `~numpy.ndarray`
  instead of array-like and don't put this line::

        data = np.asanyarray(data)

* **Do always add an `else` clause to your `if`-`elif` clauses** ... this is boilerplate code,
  but not adding it would mean users get this error if they pass an invalid option::

      UnboundLocalError: local variable 'out' referenced before assignment


Now if you really want, you can add the `numpy.asanyarray` and `isinstance` checks
for functions that end-users might often use for interactive work to provide them with
better exception messages, but doing it everywhere would mean 1000s of lines of boilerplate
code and take the fun out of Python programming.

Float data type: 32 bit or 64 bit?
----------------------------------

Most of the time what we want is to use 32 bit to store data on disk and 64 bit to do
computations in memory.

Using 64 bit to store data and results (e.g. large images or cubes) on disk would mean
a factor ~2 increase in file sizes and slower I/O, but I'm not aware of any case
where we need that precision.

On the other hand, doing computations with millions and billions of pixels very frequently
results in inaccurate results ... e.g. the likelihood is the sum over per-pixel likelihoods
and using 32-bit will usually result in erratic and hard-to-debug optimizer behaviour
and even if the fit works incorrect results.

Now you shouldn't put this line at the top of every function ... assume the caller
passes 64-bit data::

        data = np.asanyarray(data, dtype='float64')

But you should add explicit type conversions to 64 bit when reading float data from files
and explicit type conversions to 32 bit before writing to file.

Clobber or overwrite?
---------------------

In Gammapy we use on ``overwrite`` bool option for `gammapy.scripts` and functions that
write to files.

Why not use ``clobber`` instead?
After all the `FTOOLS`_ always use ``clobber``.

The reason is that ``overwrite`` is clear to everyone, but ``clobber`` is defined by the dictionary
(e.g. see `here <http://dictionary.reference.com/browse/clobber>`__)
as "to batter severely; strike heavily. to defeat decisively. to denounce or criticize vigorously."
and isn't intuitively clear to new users.

Astropy has started the process of changing their APIs to consistently use ``overwrite``
and deprecated the use of ``clobber``. So we do the same in Gammapy.

Pixel coordinate convention
---------------------------

All code in Gammapy should follow the Astropy pixel coordinate convention that the center of the first pixel
has pixel coordinates ``(0, 0)`` (and not ``(1, 1)`` as shown e.g. in ds9).
It's currently documented `here <http://photutils.readthedocs.io/en/latest/photutils/overview.html#coordinate-conventions>`__
but I plan to document it in the Astropy docs soon (see `issue 2607 <https://github.com/astropy/astropy/issues/2607>`__).

You should use ``origin=0`` when calling any of the pixel to world or world to pixel coordinate transformations in `astropy.wcs`.

When to use C or Cython or Numba for speed
------------------------------------------

Most of Gammapy is written using Python and Numpy array expressions calling functions (e.g. from Scipy)
that operate on Numpy arrays.
This is often nice because it means that algorithms can be implemented with few lines of high-level code,

There is a very small fraction of code though (one or a few percent) where this results in code that is
either cumbersome or too slow. E.g. to compute TS or upper limit images, one needs to do a root finding
method with different number of iterations for each pixel ... that's impossible (or at least very
cumbersome / hard to read) to implement with array expressions and Python loops over pixels are slow.

In these cases we encourage the use of `Cython <http://cython.org/>`__ or `Numba <http://numba.pydata.org/>`__,
or writing the core code in C and exposing it to Python via Cython.
These are popular and simple ways to get C speed from Python.

To use several CPU cores consider using the Python standard library
`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`__ module.

Note that especially the use of Numba should be considered an experiment.
It is a very nice, but new technology and no-one uses it in production.
Before the Gammapy 1.0 release we will re-evaluate the status of Numba and decide whether it's
an optional dependency we use for speed, or whether we use the much more established Cython
(Scipy, scikit-image, Astropy, ... all use Cython).

At the time of writing (April 2015), the TS map computation code uses Cython and multiprocessing
and Numba is not used yet.

What belongs in Gammapy and what doesn't?
-----------------------------------------

The scope of Gammapy is currently not very well defined ... if in doubt whether it makes sense to
add something, please ask on the mailing list or via a Github issue.

Roughly the scope is high-level science analysis of gamma-ray data, starting with event lists
after gamma-hadron separation and corresponding IRFs, as well as source and source population modeling.

For lower-level data processing (calibration, event reconstruction, gamma-hadron separation)
there's `ctapipe`_. There's some functionality (event list processing, PSF or background model building,
sensitivity computations ...) that could go in either ctapipe or Gammapy and we'll have to try
and avoid duplication.

SED modeling code belongs in `naima`_.

A lot of code that's not gamma-ray specific belongs in other packages
(e.g. `Scipy`_, `Astropy`_, other Astropy-affiliated packages, `Sherpa`_).
We currently have quite a bit of code that should be moved "upstream" or already has been,
but the Gammapy code hasn't been adapted yet.

Assert convention
-----------------

When performing tests, the preferred numerical assert method is
`numpy.testing.assert_allclose`. Use

.. code-block:: python

    from numpy.testing import assert_allclose

at the top of the file and then just use ``assert_allclose`` for
the tests. This makes the lines shorter, i.e. there is more space
for the arguments.

``assert_allclose`` covers all use cases for numerical asserts, so
it should be used consistently everywhere instead of using the
dozens of other available asserts from pytest or numpy in various
places.

In case of assertion on arrays of quantity objects, such as
`~astropy.units.Quantity` or `~astropy.coordinates.Angle`, the
following method can be used:
`astropy.tests.helper.assert_quantity_allclose`.
In this case, use

.. code-block:: python

    from astropy.tests.helper import assert_quantity_allclose

at the top of the file and then just use ``assert_quantity_allclose``
for the tests.

.. _dev_random:

Random numbers
--------------

All functions that need to call a random number generator should
take a ``random_state`` input parameter and call the
`~gammapy.utils.random.get_random_state` utility function like this
(you can copy & paste the three docstring lines and the first code line
to the function you're writing):

.. code-block:: python

    from gammapy.utils.random import get_random_state

    def make_random_stuff(X, random_state='random-seed'):
        """...

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        random_state = get_random_state(random_state)
        data = random_state.uniform(low=0, high=3, size=10)
        return data

This allows callers flexible control over which random number generator
(i.e. which `numpy.random.RandomState` instance) is used and how it's initialised.
The default ``random_state='random-seed'`` means "create a new RNG, seed it in a random way",
i.e. different random numbers will be generated on every call.

There's a few ways to get deterministic results from a script that call
functions that generate random numbers.

One option is to create a single `~numpy.random.RandomState` object seeded with an integer
and then pass that ``random_state`` object to every function that generates random numbers:

.. code-block:: python

    from numpy.random import RandomState
    random_state = RandomState(seed=0)

    stuff1 = make_some_random_stuff(random_state=random_state)
    stuff2 = make_more_random_stuff(random_state=random_state)


Another option is to pass an integer seed to every function that generates random numbers:

.. code-block:: python

    seed = 0
    stuff1 = make_some_random_stuff(random_state=seed)
    stuff2 = make_more_random_stuff(random_state=seed)

This pattern was inspired by the way
`scikit-learn handles random numbers <http://scikit-learn.org/stable/developers/#random-numbers>`__.
We have changed the ``None`` option of `sklearn.utils.check_random_state` to ``'global-rng'``,
because we felt that this meaning for ``None`` was confusing given that `numpy.random.RandomState`
uses a different meaning (for which we use the option ``'global-rng'``).

Logging
-------

Gammapy is a library. This means that it should never contain print statements, because with
print statements the library users have no easy way to configure where the print output goes
(e.g. to ``stdout`` or ``stderr`` or a log file) and what the log level (``warning``, ``info``, ``debug``)
and format is (e.g. include timestamp and log level?).

So logging is much better than printing. But also logging is only rarely needed.
Many developers use print or log statements to debug some piece of code while they write it.
Once it's written and works, it's rare that callers want it to be chatty and log messages all the time.
Print and log statements should mostly be contained in end-user scripts that use Gammapy,
not in Gammapy itself.

That said, there are cases where emitting log messages can be useful.
E.g. a long-running algorithm with many steps can log info or debug statements.
In a function that reads and writes several files it can make sense to include info log messages
for normal operation, and warning or error log messages when something goes wrong.
Also, command line tools that are included in Gammapy **should** contain log messages,
informing the user about what they are doing.

Gammapy uses the Python standard library `logging` module. This module is extremely flexible,
but also quite complex. But our logging needs are very modest, so it's actually quite simple ...

Generating log messages
+++++++++++++++++++++++

To generate log messages from any file in Gammapy, include these two lines at the top:

.. code-block:: python

    import logging
    log = logging.getLogger(__name__)

This creates a module-level `logging.Logger` object called ``log``, and you can then create
log messages like this from any function or method:

.. code-block:: python

    def process_lots_of_data(infile, outfile):

        log.info('Starting processing data ...')

        # do lots of work

        log.info('Writing {}'.format(outfile))


You should never log messages from the module level (i.e. on import) or configure the log
level or format in Gammapy, that should be left to callers ... except from command line tools ...

There is also the rare case of functions or classes with the main job to check
and log things. For these you can optionally let the caller pass a logger when
constructing the class to make it easier to configure the logging.
See the `~gammapy.data.EventListDatasetChecker` as an example.

Configuring logging from command line tools
+++++++++++++++++++++++++++++++++++++++++++

Every Gammapy command line tool should have a ``--loglevel`` option:

.. code-block:: python

    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")

This option is then processed at the end of ``main`` using this helper function:

.. code-block:: python

    set_up_logging_from_args(args)

This sets up the root logger with the log level and format (the format isn't configurable
for the command line scripts at the moment).

See ``gammapy/scripts/find_obs.py`` as an example.


Command line tools using click
------------------------------

Command line tools that use the `click <http://click.pocoo.org/>`_ module should disable
the unicode literals warnings to clean up the output of the tool:

.. code-block:: python

    import click
    click.disable_unicode_literals_warning = True

See `here <http://click.pocoo.org/5/python3/#unicode-literals>`_ for further
information.


BSD or GPL license?
-------------------

Gammapy is BSD licensed (same license as Numpy, Scipy, Matplotlib, scikit-image, Astropy, photutils, yt, ...).

We prefer this over the GPL3 or LGPL license because it means that the packages we are most likely to
share code with have the same license, e.g. we can take a function or class and "upstream" it, i.e. contribute
it e.g. to Astropy or Scipy if it's generally useful.

Some optional dependencies of Gammapy (i.e. other packages like Sherpa or Gammalib or ROOT that we import in some
places) are GPL3 or LGPL licensed.

Now the GPL3 and LGPL license contains clauses that other package that copy or modify it must be released under
the same license.
We take the standpoint that Gammapy is independent from these libraries, because we don't copy or modify them.
This is a common standpoint, e.g. ``astropy.wcs`` is BSD licensed, but uses the LGPL-licensed WCSLib.

Note that if you distribute Gammapy together with one of the GPL dependencies,
the whole distribution then falls under the GPL license.

Changelog
---------

In Gammapy we keep a :ref:`changelog` with a list of pull requests.
We sort by release and within the release by PR number (largest first).

As explained in the :ref:`astropy:changelog-format` section in the Astropy docs,
there are (at least) two approaches for adding to the changelog, each with pros
and cons.

We've had some pain due to merge conflicts in the changelog and having to wait
until the contributor rebases (and having to explain git rebase to new contributors).

So our recommendation is that changelog entries are not added in pull requests,
but that the core developer adds a changelog entry after right after having
merged a pull request (you can add ``[skip ci]`` on this commit).

File and directory path handling
--------------------------------

In Gammapy use ``Path`` objects to handle file and directory paths.

.. code-block:: python

    from gammapy.extern.pathlib import Path

    dir = Path('folder/subfolder')
    filename = dir / 'filename.fits'
    dir.mkdir(exist_ok=True)
    table.write(str(filename))

Note how the ``/`` operator makes it easy to construct paths
(as opposed to repeated calls to the string-handling function ``os.path.join``)
and how methods on ``Path`` objects provide a nicer interface to most
of the functionality from ``os.path`` (``mkdir`` in this example).

One gotcha is that many functions (such as ``table.write`` in this example)
expect ``str`` objects and refuse to work with ``Path`` objects, so you have
to explicitly convert to ``str(path)``.

Note that pathlib was added to the Python standard library in 3.4
(see `here <https://docs.python.org/3/library/pathlib.html>`__),
but since we support Python 2.7 and the Python devs keep improving the
version in the standard library (by adding new methods and new options
for existing methods), we decided to bundle the latest version
(from `here <https://pypi.python.org/pypi/pathlib2/>`__) in
``gammapy/extern/pathlib.py`` and that should always be used.

Bundled gammapy.extern code
---------------------------

We bundle some code in ``gammapy.extern``.
This is external code that we don't maintain or modify in Gammapy.
We only bundle small pure-Python files (currently all single-file modules) purely for convenience,
because having to explain about these modules as Gammapy dependencies to end-users would be annoying.
And in some cases the file was extracted from some other project, i.e. can't be installed
separately as a dependency.

For ``gammapy.extern`` we don't generate Sphinx API docs.
To see what is there, check out the ``gammapy/extern`` directory locally or on
`Github <https://github.com/gammapy/gammapy/tree/master/gammapy/extern>`__.
Notes on the bundled files are kept in the docstring of
`gammapy/extern/__init__.py <https://github.com/gammapy/gammapy/blob/master/gammapy/extern/__init__.py>`__.

.. _interpolation-extrapolation:

Interpolation and extrapolation
-------------------------------

In Gammapy, we use interpolation a lot, e.g. to evaluate instrument response functions (IRFs) on
data grids, or to reproject diffuse models on data grids.

Note: For some use cases that require interpolation the
`~gammapy.utils.nddata.NDDataArray` base class might be useful.

The default interpolator we use is `scipy.interpolate.RegularGridInterpolator` because it's fast and robust
(more fancy interpolation schemes can lead to unstable response in some cases, so more careful checking
across all of parameter space would be needed).

You should use this pattern to implement a function of method that does interpolation:

.. code-block:: python

    def do_something(..., interp_kwargs=None):
        """Do something.

        Parameters
        ----------
        interp_kwargs : dict or None
            Interpolation parameter dict passed to `scipy.interpolate.RegularGridInterpolator`.
            If you pass ``None``, the default ``interp_params=dict(bounds_error=False)`` is used.
        """
        if not interp_kwargs:
            interp_kwargs = dict(bounds_error=False)

        interpolator = RegularGridInterpolator(..., **interp_kwargs)

Since the other defaults are ``method='linear'`` and ``fill_value=nan``, this implies that linear interpolation
is used and `NaN`_ values are returned for points outside of the interpolation domain.
This is a compromise between the alternatives:

* ``bounds_error=True`` -- Very "safe", refuse to return results for any points if one of the points is outside the valid domain.
  Can be annoying for the caller to not get any result.
* ``bounds_error=False, fill_value=nan`` -- Medium "safe". Always return a result, but put NaN values to make it easy
  for analysers to spot that there's an issue in their results (if pixels with NaN are used, that will usually lead
  to NaN values in high-level analysis results.
* ``bounds_error=False, fill_value=0`` or ``bounds_error=False, fill_value=None`` -- Least "safe".
  Extrapolate with zero or edge values (this is what ``None`` means).
  Can be very convenient for the caller, but can also lead to errors where e.g. stacked high-level analysis results
  aren't quite correct because IRFs or background models or ... were used outside their valid range.

Methods that use interpolation should provide an option to the caller to pass interpolation options on to
``RegularGridInterpolator`` in case the default behaviour doesn't suit the application.

TODO: we have some classes (aeff2d and edisp2d) that pre-compute an interpolator, currently in the constructor.
In those cases the ``interp_kwargs`` would have to be exposed e.g. also on the `read` and other constructors.
Do we want / need that?


Locate origin of warnings
-------------------------

By default, warnings appear on the console, but often it's not clear where a given warning
originates (e.g. when building the docs or running scripts or tests) or how to fix it.

Sometimes putting this in ``gammapy/__init__.py`` can help::

    import numpy as np
    np.seterr(all='raise')

Following the advice `here <http://stackoverflow.com/questions/22373927/get-traceback-of-warnings/22376126#22376126>`__,
putting this in ``docs/conf.py`` can also help sometimes::

    import traceback
    import warnings
    import sys

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        traceback.print_stack()
        log = file if hasattr(file,'write') else sys.stderr
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback


Object summary info string
--------------------------

If you want to add a method to provide some basic information about a class instance,
you should use the Python ``__str__`` method.

.. code-block:: python

    class Spam(object):
        def __init__(self, ham):
            self.ham = ham

        def __str__(self):
            ss = 'Summary Info about class Spam\n'
            ss += '{:.2f}'.format(self.ham)
            return ss

If you want to add configurable info output, please provide a method ``summary``,
like :func:`here <gammapy.catalog.SourceCatalogObjectHGPS.summary>`.
In this case the ``__str__`` method should be a call to ``summary`` with default
parameters. Do not use an ``info`` method, since this would lead to conflicts
for some classes in Gammapy (e.g. classes that inherit the ``info`` method from
``astropy.table.Table``.


Validating H.E.S.S. FITS exporters
----------------------------------

The H.E.S.S. experiment has 3 independent analysis chains, which all have exporters to the :ref:`gadf:main-page` format.
The Gammapy tests contain a mechanism to track changes in these exporters.


In the ``gammapy-extra`` repository there is a script ``test_datasets/reference/make_reference_files.py`` that reads
IRF files from different chains and prints the output of the ``__str__`` method to a file. It also creates a YAML file
holding information about the datastore used for each chain, the observations used, etc.


The test ``gammapy/irf/tests/test_hess_chains.py`` load exactly the same files as the script and compares the output of the
``__str__`` function to the reference files on disk. That way all changes in the exporters or the way the IRF files are read by
Gammapy can be tracked. So, if you made changes to the H.E.S.S. IRF exporters you have to run the ``make_reference_files.py`` script
again to ensure the passing of all Gammapy tests.


If you want to compare the IRF files between two different datastores (to compare between to chains or fits productions) you have to
 manually edit the YAML file written by ``make_reference_files.py`` and include the info which datastore should be compared to which reference file.


.. _use-nddata:

Using the NDDataArray
---------------------

Gammapy has a class for generic n-dimensional data arrays,
`~gammapy.utils.nddata.NDDataArray`. Classes that represent such an array
should use this class. The goal is to reuse code for interpolation
and have an coherent I/O interface, mainly in `~gammapy.irf`.

A usage example can be found in :gp-extra-notebooks:``nddata_demo``.

Also, consult :ref:`interpolation-extrapolation` if you are not sure how to
setup your interpolator.


Write a test for an IPython notebook
------------------------------------

There is a script called ``test_notebooks.py`` in the gammapy main folder. It
exectues all notebooks listed in file ``notebook.yaml`` in
``gammapy-extra/notebooks.yaml`` using
`runipy <https://github.com/paulgb/runipy>`__. So if you edit an existing
notebook or make changes to gammapy that break an existing notebook, you have
to run ``test_notebooks.py`` until all notebooks run without raising an error.
If you add a new notebook and want it to be under test (which of course is what
you want) you have to add it to ``gammapy-extra/notebooks/notebooks.yaml``.
Note that there is also the command ``make test-notebooks`` which is used for

continuous integration on travis CI. It is not recommended to use this locally,
since it overwrides your gammapy installation (see issue 727).

Sphinx docs build
-----------------

Generating the HTML docs for Gammapy is straight-forward::

    python setup.py build_docs
    open docs/_build/html/index.html

Generating the PDF docs is more complex.
This should work::

    python setup.py build_docs -b latex
    cd docs/_build/latex
    makeindex -s python.ist gammapy.idx
    pdflatex -interaction=nonstopmode gammapy.tex
    open gammapy.pdf

You need a bunch or LaTeX stuff, specifically ``texlive-fonts-extra`` is needed.

The PDF is also generated on Read the Docs and available online here:
https://media.readthedocs.org/pdf/gammapy/latest/gammapy.pdf

Jupyter notebooks present in the ``gammapy-extra`` repository are by default copied
to the ``docs/notebooks`` and ``docs/_static/notebooks`` tree-folder structure during
the process of generating HTML docs. This triggers its conversion to *fixed-text*
Sphinx formatted documentation files and at the same time provides access to raw
.ipynb Jupyter notebooks for the same version of the gammapy documentation. This
behaviour may be modified in the  `setup.cfg` configuration file changing the
value of `clean_notebooks` boolean.

Documentation guidelines
------------------------

Like almost all Python projects, the Gammapy documentation is written in a format called
`restructured text (RST)`_ and built using `Sphinx`_.
We mostly follow the :ref:`Astropy documentation guidelines <astropy:documentation-guidelines>`,
which are based on the `Numpy docstring standard`_,
which is what most scientific Python packages use.

.. _restructured text (RST) : http://sphinx-doc.org/rest.html
.. _Sphinx: http://sphinx-doc.org/
.. _Numpy docstring standard: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

There's a few details that are not easy to figure out by browsing the Numpy or Astropy
documentation guidelines, or that we actually do differently in Gammapy.
These are listed here so that Gammapy developers have a reference.

Usually the quickest way to figure out how something should be done is to browse the Astropy
or Gammapy code a bit (either locally with your editor or online on Github or via the HTML docs),
or search the Numpy or Astropy documentation guidelines mentioned above.
If that doesn't quickly turn up something useful, please ask by putting a comment on the issue or
pull request you're working on on Github, or send an email to the Gammapy mailing list.

Functions or class methods that return a single object
++++++++++++++++++++++++++++++++++++++++++++++++++++++

For functions or class methods that return a single object, following the
Numpy docstring standard and adding a *Returns* section usually means
that you duplicate the one-line description and repeat the function name as
return variable name.
See `astropy.cosmology.LambdaCDM.w` or `astropy.time.Time.sidereal_time`
as examples in the Astropy codebase. Here's a simple example:

.. code-block:: python

    def circle_area(radius):
        """Circle area.

        Parameters
        ----------
        radius : `~astropy.units.Quantity`
            Circle radius

        Returns
        -------
        area : `~astropy.units.Quantity`
            Circle area
        """
        return 3.14 * (radius ** 2)

In these cases, the following shorter format omitting the *Returns* section is recommended:

.. code-block:: python

    def circle_area(radius):
        """Circle area (`~astropy.units.Quantity`).

        Parameters
        ----------
        radius : `~astropy.units.Quantity`
            Circle radius
        """
        return 3.14 * (radius ** 2)

Usually the parameter description doesn't fit on the one line, so it's
recommended to always keep this in the *Parameters* section.

This is just a recommendation, e.g. for `gammapy.cube.SkyCube.spectral_index`
we decided to use this shorter format, but for `gammapy.cube.SkyCube.flux` we
decided to stick with the more verbose format, because the return type and units
didn't fit on the first line.

A common case where the short format is appropriate are class properties,
because they always return a single object.
As an example see `gammapy.data.EventList.radec`, which is reproduced here:

.. code-block:: python

    @property
    def radec(self):
        """Event RA / DEC sky coordinates (`~astropy.coordinates.SkyCoord`).
        """
        lon, lat = self['RA'], self['DEC']
        return SkyCoord(lon, lat, unit='deg', frame='icrs')


Class attributes
++++++++++++++++

Class attributes (data members) and properties are currently a bit of a mess,
see `~gammapy.cube.SkyCube` as an example.
Attributes are listed in an *Attributes* section because I've listed them in a class-level
docstring attributes section as recommended `here`__.
Properties are listed in separate *Attributes summary* and *Attributes Documentation*
sections, which is confusing to users ("what's the difference between attributes and properties?").

.. __: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#class-docstring

One solution is to always use properties, but that can get very verbose if we have to write
so many getters and setters. I don't have a solution for this yet ... for now I'll go read
`this`__ and meditate.

.. __: https://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb

TODO: make a decision on this and describe the issue / solution here.

Constructor parameters
++++++++++++++++++++++

TODO: should we put the constructor parameters in the class or ``__init__`` docstring?

Different versions of notebooks in Binder
-----------------------------------------

Jupyter notebooks may be accessed and executed on-line in the
`Gammapy Binder <http://mybinder.org/repo/gammapy/gammapy-extra>`__ space. Each *fixed-text* sphinx
formatted notebook present in the documentation has its own link pointing to its specific space in
Gammapy Binder. Since notebooks are evolving with Gammapy functionalities and documentation, it is
possible to link the different versions of the notebooks stored in GitHub repository ``gammapy-extra``
to the same versions built in Gammapy Binder. For this purpose just edit the variable **git_commit**
in ``setup.cfg`` file and provide the branch, tag or commit of GitHub repository ``gammapy-extra``
that will be used to access the same version of the notebook in Gammapy Binder.

Link to a notebook in gammapy-extra from the docs
-------------------------------------------------

Jupyter notebooks stored in ``gammpy-extra`` are copied to thew ``notebooks`` folder
during the process of Sphinx building documentation. They are converted to HTML files
using `nb_sphinx <http://nbsphinx.readthedocs.io/>`__ Sphinx extension that provides
a source parser for .ipynb files.

From docstrings and high-level docs in Gammapy, you can link to these *fixed-text*
formatted version of the notebooks **providing the relative path to** ``notebooks`` **folder and .html file extension**:

Example: `First steps with Gammapy <../notebooks/first_steps.html>`__

Sphinx directive to generate that link::

    `First steps with Gammapy <../notebooks/first_steps.html>`__

If you want to link to notebooks rendered on the external
**NBViewer platform** you can use the ``gp-extra-notebook``
Sphinx role providing **only the filename**.

Example: :gp-extra-notebook:`image_analysis`

Sphinx directive to generate that link::

      :gp-extra-notebook:`image_analysis`

More info on Sphinx roles is `here <http://www.sphinx-doc.org/en/stable/markup/inline.html>`__

Include images from gammapy-extra into the docs
-----------------------------------------------

Similar to the ``gp-extra-notebook`` role, Gammapy has a ``gp-extra-image`` directive.

To include an image from ``gammapy-extra/figures/``, use the ``gp-extra-image`` directive
instead of the usual Sphinx ``image`` directive like this:


.. code-block:: rst

    .. gp-extra-image:: detect/fermi_ts_image.png
        :scale: 100%

More info on the image directive is `here <http://www.sphinx-doc.org/en/stable/rest.html#images>`__

.. _dev-wipe_rtd:

Wipe readthedocs
----------------

After things (classes, methods, functions) are removed, the Sphinx API docs often show these old items.
If you notice this, you have to "wipe" the Gammapy install on Readthedocs and start a fresh build.
If you don't have permissions on Readthedocs, file a Github issue or mention this on the mailing list.

The wipe procedure is described `here <http://read-the-docs.readthedocs.io/en/latest/builds.html#deleting-a-stale-or-broken-build-environment>`__.

The steps are:

* log in `here <https://readthedocs.org/accounts/login/>`__
* hit this URL and click the "wipe" button to wipe the existing install:

   https://readthedocs.org/wipe/gammapy/latest/
* go `here <https://readthedocs.org/projects/gammapy/>`__ and clicking the "Build" button.
* go `here <https://readthedocs.org/builds/gammapy/>`__ and check if the build succeeded
* re-check the output docs page where you had previously seen something outdated.
