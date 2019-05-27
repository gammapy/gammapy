.. include:: ../references.txt

.. _dev_howto:

***************
Developer HOWTO
***************

This page is a collection of notes for Gammapy contributors and maintainers,
in the form of short "How to" or "Q & A" entries.

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

It is up to the developer to decide how to return multiple things from functions and methods.
For up to three things, if callers usually will want access to several things,
using a ``tuple`` or ``collections.namedtuple`` is OK.
For three or more things, using a Python ``dict`` instead should be preferred.

.. _dev-python2and3:

Python version support
----------------------

In Gammapy we currently support Python 3.5 or later.

We plan to discuss later in 2019 whether to bump the version requirement to Python 3.6,
to be able to take advantage of the new features introduced there.

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

In Gammapy we consistently use an ``overwrite`` bool option for `gammapy.scripts` and functions that
write to files. This is in line with Astropy, which had a mix of ``clobber`` and ``overwrite`` in
the past, and has switched to uniform ``overwrite`` everywhere.

The default value should be ``overwrite=False``, although we note that this decision was very
controversial, several core developers would prefer to use ``overwrite=True``.
For discussion on this, see `GH 1396 <https://github.com/gammapy/gammapy/issues/1396>`__.

Pixel coordinate convention
---------------------------

All code in Gammapy should follow the Astropy pixel coordinate convention that the center of the first pixel
has pixel coordinates ``(0, 0)`` (and not ``(1, 1)`` as shown e.g. in ds9).

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

For assertions on `~astropy.units.Quantity` objects, you can do this
to assert on the unit and value separately:

.. code-block:: python

    from numpy.testing import assert_allclose
    import astropy.units as u

    actual = 1 / 3 * u.deg
    assert actual.unit == 'deg'
    assert_allclose(actual.value, 0.33333333)

Note that  `~astropy.units.Quantity` can be compared to unit strings directly.
Also note that the default for ``assert_allclose`` is ``atol=0`` and ``rtol=1e-7``,
so when using it, you have to give the reference value with a precision of
``rtol ~ 1e-8``, i.e. 8 digits to be on the safe side (or pass a lower ``rtol`` or set an ``atol``).

The use of `~astropy.tests.helper.assert_quantity_allclose` is discouraged,
because it only requires that the values match after unit conversions.
This is not so bad, but units in test cases should not change randomly,
so asserting on unit and value separately establishes more behaviour.

If you don't like the two separate lines, you can use `gammapy.utils.testing.assert_quantity_allclose`,
which does assert that units are equal, and calls `numpy.testing.assert_equal` for the values.

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
We have changed the ``None`` option of ``sklearn.utils.check_random_state`` to ``'global-rng'``,
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

In Gammapy use `pathlib.Path` objects to handle file and directory paths.

.. code-block:: python

    from pathlib import Path

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
In those cases the ``interp_kwargs`` would have to be exposed e.g. also on the ``read`` and other constructors.
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


Object text repr, str and info
------------------------------

In Python, by default objects don't have a good string representation. This
section explains how Python repr, str and print work, and gives guidelines for
writing ``__repr__``, ``__str__`` and ``info`` methods on Gammapy classes.

Let's use this as an example::

    class Person:
        def __init__(self, name='Anna', age=8):
            self.name = name
            self.age = age

The default ``repr`` and ``str`` are this::

    >>> repr(p)
    '<__main__.Person object at 0x105fe3b70>'
    >>> p.__repr__()
    '<__main__.Person object at 0x105fe3b70>'
    >>> str(p)
    '<__main__.Person object at 0x105fe3b70>'
    >>> p.__str__()

Users will see that. If they just give an object in the Python REPL, the
``repr`` is shown. If they print the object, the ``str`` is shown. In both cases
without the quotes seen above.

    >>> p = Person()
    >>> p
    <__main__.Person at 0x105fd0cf8>
    >>> print(p)
    <__main__.Person object at 0x105fe3b70>

There are ways to make this better and avoid writing boilerplate code,
specifically `attrs <http://www.attrs.org/>`__ and `dataclasses
<https://docs.python.org/3/library/dataclasses.html>`__. We might use those in
the future in Gammapy, but for now, we don't.

If you want a better repr or str for a given object, you have to add
``__repr__`` and / or ``__str__`` methods when writing the class. Note that you
don't have to do that, it's mainly useful for objects users interact with a lot.
For classes that are mainly used internally, developers can e.g. just do this to
see the attributes printed nicely::

    >>> p.__dict__
    {'name': 'Anna', 'age': 8}


Here's an example how to write ``__repr__``::

    def __repr__(self):
        return '{}(name={!r}, age={!r})'.format(
            self.__class__.__name__, self.name, self.age
        )

Note how we use ``{!r}`` in the format string to fill in the ``repr`` of the
object being formatted, and how we used ``self.__class__.__name__`` to avoid
duplicating the class name (easier to refactor code, and shows sub-class name if
repr is inherited).

This will give a nice string representation. The same one for ``repr`` and
``str``, you don't have to write ``__str__``::

    >>> p = Person(name='Anna', age=8)
    >>> p
    Person(name='Anna', age=8)
    >>> print(p)
    Person(name='Anna', age=8)

The string representation is usually used for more informal or longer printout.
Here's an example::

    def __str__(self):
        return (
            "Hi, my name is {} and I'm {} years old.\n"
            "I live in Heidelberg."
        ).format(self.name, self.age)

If you need text representation that is configurable, i.e. tables arguments what
to show, you should add a method called ``info``. To avoid code duplication, you
should then call ``info`` from ``__str__``. Example::

    class Person:
        def __init__(self, name='Anna', age=8):
            self.name = name
            self.age = age

        def __repr__(self):
            return '{}(name={!r}, age={!r})'.format(
                self.__class__.__name__, self.name, self.age
            )

        def __str__(self):
            return self.info(add_location=False)

        def info(self, add_location=True):
            s = ("Hi, my name is {} and I'm {} years old."
                ).format(self.name, self.age)
            if add_location:
                s += "\nI live in Heidelberg"
            return s

This pattern of returning a string from ``info`` has some pros and cons.
It's easy to get the string, and do what you like with it, e.g. combine
it with other text, or store it in a list and write it to file later.
The main con is that users have to call ``print(p.info())`` to see a
nice printed version of the string instead of ``\n``::

    >>> p = Person()
    >>> p.info()
    "Hi, my name is Anna and I'm 8 years old.\nI live in Heidelberg"
    >>> print(p.info())
    Hi, my name is Anna and I'm 8 years old.
    I live in Heidelberg

To make ``info`` print by default, and be re-usable from ``__str__`` and make it
possible to get a string (without having to monkey-patch ``sys.stdout``), would
require adding this ``show`` option and if-else at the end of every ``info``
method::

    def __str__(self):
        return self.info(add_location=False, show=False)

    def info(self, add_location=True, show=True):
        s = ("Hi, my name is {} and I'm {} years old."
             ).format(self.name, self.age)
        if add_location:
            s += "\nI live in Heidelberg"

        if show:
            print(s)
        else:
            return s

To summarise: start without adding and code for text representation. If there's a
useful short text representation, you can add a ``__repr__``. If really useful,
add a ``__str__``. If you need it configurable, add an ``info`` and call
``info`` from ``str``. If ``repr`` and ``str`` are similar, it's not really
useful: delete the ``__str__`` and only keep the ``__repr__``.

It is common to have bugs in ``__repr__``, ``__str__`` and ``info`` that are not
tested. E.g. a ``NameError`` or ``AttributeError`` because some attribute name
changed, and updating the repr / str / info was forgotten. So tests should be added
that execute these methods once. You can write the reference string in the output,
but that is not required (and actually very hard for cases where you have floats
or Numpy arrays or str, where formatting differs across Python or Numpy version.
Example what to put as a test::

    def test_person_txt():
        p = Person()
        assert repr(p).startswith('Person')
        assert str(p).startswith('Hi')
        assert p.info(add_location=True).endswith('Heidelberg')

.. _use-nddata:

Using the NDDataArray
---------------------

Gammapy has a class for generic n-dimensional data arrays,
`~gammapy.utils.nddata.NDDataArray`. Classes that represent such an array
should use this class. The goal is to reuse code for interpolation
and have an coherent I/O interface, mainly in `~gammapy.irf`.

Also, consult :ref:`interpolation-extrapolation` if you are not sure how to
setup your interpolator.

Sphinx docs build
-----------------

Generating the HTML docs for Gammapy is straight-forward::

    make docs-all
    make docs-show

Generating the PDF docs is more complex.
This should work::

    python setup.py build_docs -b latex
    cd docs/_build/latex
    makeindex -s python.ist gammapy.idx
    pdflatex -interaction=nonstopmode gammapy.tex
    open gammapy.pdf

You need a bunch or LaTeX stuff, specifically ``texlive-fonts-extra`` is needed.

Jupyter notebooks stripped of output cells are present in the ``tutorials`` folder.
They are by default tested, executed, and copied to the ``docs/notebooks`` and
``docs/_static/notebooks`` folders during the process of generating HTML docs. This
triggers its conversion to Sphinx formatted HTML files and ``.py`` scripts. The Sphinx
formatted versions of the notebooks provide links to the raw ``.ipynb`` Jupyter files
and ``.py`` script versions stored in ``docs/_static/notebooks`` folder.

Once the documentation built you can optimize the speed of re-building processes,
for example in case you are modifying or creating new docs and you would like to check
these changes are displayed nicely. For that purpose, if your modified RST file
does not contain links to notebooks, you may run ``make docs-all nbs=False`` so
that notebooks are not executed during the docs build.

In the case one single notebook is modified or added to the documentation, you can
execute the build doc process with the ``src`` parameter with value the name of the
considered notebook. i.e. ``make docs-all src=tutorials/my-notebook.ipynb``

Each *fixed-text* Sphinx formatted notebook present in the documentation has its
own link pointing to its specific space in Gammapy Binder. Since notebooks are
evolving with Gammapy functionalities and documentation, it is possible to link
the different versions of the notebooks to the same versions built in Gammapy Binder.
For stable releases, it is useful to use the ``release`` parameter with the value
of the release label tag used in Github. This value will be used to build the links
to Binder for that specific stable release for each of the tutorials published in
the :ref:`tutorials` section. i.e. ``make docs-all release=v0.8``

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

Class attributes (data members) and properties are currently a bit of a mess.
Attributes are listed in an *Attributes* section because I've listed them in a class-level
docstring attributes section as recommended `here`__.
Properties are listed in separate *Attributes summary* and *Attributes Documentation*
sections, which is confusing to users ("what's the difference between attributes and properties?").

.. __: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#class-docstring

One solution is to always use properties, but that can get very verbose if we have to write
so many getters and setters. We could start using descriptors.

TODO: make a decision on this and describe the issue / solution here.

Link to a notebook from the docs
-------------------------------------------------

Jupyter notebooks stored in the ``tutorials`` folder and are copied to the ``notebooks`` folder
during the process of Sphinx building documentation. They are converted to HTML files using
`nb_sphinx <http://nbsphinx.readthedocs.io/>`__ Sphinx extension that provides a source parser
for .ipynb files. From docstrings and high-level docs in Gammapy you can link to these
*fixed-text* formatted versions using the ``gp-notebook`` Sphinx role providing
**only the filename**.

Example: :gp-notebook:`analysis_3d`

Sphinx directive to generate that link::

      :gp-notebook:`analysis_3d`

More info on Sphinx roles is `here <http://www.sphinx-doc.org/en/stable/markup/inline.html>`__

Alternatively you can also link to the notebooks providing its filename with .html file extension and the relative path to the ``notebooks`` folder. This folder is created at the root of the ``docs`` folder in the process of documentation building.

Example: `First steps with Gammapy <../notebooks/first_steps.html>`__

Sphinx directive to generate that link::

    `First steps with Gammapy <../notebooks/first_steps.html>`__


Include images from gammapy-extra into the docs
-----------------------------------------------

Similar to the ``gp-notebook`` role, Gammapy has a ``gp-image`` directive.

To include an image from ``gammapy-extra/figures/``, use the ``gp-image`` directive
instead of the usual Sphinx ``image`` directive like this:


.. code-block:: rst

    .. gp-image:: detect/fermi_ts_image.png
        :scale: 100%

More info on the image directive is `here <http://www.sphinx-doc.org/en/stable/rest.html#images>`__

Coordinate and axis names
-------------------------

In Gammapy, the following coordinate and axis names should be used.

This applies to most of the code, ranging from IRFs to maps
to sky models, for function parameters and variable names.

* ``time`` - time
* ``energy`` - energy
* ``ra``, ``dec`` - sky coordinates, ``radec`` frame (i.e. ``icrs`` to be precise)
* ``glon``, ``glat`` - sky coordinates, ``galactic`` frame
* ``az``, ``alt`` - sky coordinates, ``altaz`` frame
* ``lon``, ``lat`` for spherical coordinates that aren't in a specific frame.

For angular sky separation angles:

* ``psf_theta`` - offset wrt. PSF center position
* ``fov_theta`` - offset wrt. field of view (FOV) center
* ``theta`` - when no PSF is involved, e.g. to evaluate spatial sky models

For the general case of FOV coordinates that depend on angular orientation
of the FOV coordinate frame:

* ``fov_{frame}_lon``, ``fov_{frame}_lat`` - field of view coordinates
* ``fov_theta``, ``fov_{frame}_phi`` - field of view polar coordinates

where ``{frame}`` can be one of ``radec``, ``galactic`` or ``altaz``,
depending on with which frame the FOV coordinate frame is aligned.

Notes:

* In cases where it's unclear if the value is for true or reconstructed event
  parameters, a postfix ``_true`` or ``_reco`` should be added.
  In Gammapy, this mostly occurs for ``energy_true`` and ``energy_reco``,
  e.g. the background IRF has an axis ``energy_reco``, but effective area
  usually ``energy_true``, and energy dispersion has both axes.
  We are not pedantic about adding ``_true`` and ``_reco`` everywhere.
  Note that this would quickly become annoying (e.g. source models use true
  parameters, and it's not clear why one should write ``ra_true``).
  E.g. the property on the event list ``energy`` matches the ``ENERGY``
  column from the event list table, which is for real data always reco energy.
* Currently, no sky frames centered on the source, or non-radially symmetric
  PSFs are in use, and thus the case of "source frames" that have to be with
  a well-defined alignment, like we have for the "FOV frames" above,
  doesn't occur and thus doesn't need to be defined yet (but it would be natural
  to use the same naming convention as for FOV if it eventually does occur).
* These definitions are mostly in agreement with the `format spec <gadf>`_.
  We do not achieve 100% consistency everywhere in the spec and Gammapy code.
  Achieving this seems unrealistic, because legacy formats have to be supported,
  we are not starting from scratch and have time to make all formats consistent.
  Our strategy is to do renames on I/O where needed, to and from the internal
  Gammapy names defined here, to the names used in the formats.
  Of course, where formats are not set in stone yet, we advocate and encourage
  the use of the names chosen here.
* Finally, we realise that eventually probably CTA will define this, and Gammapy
  is only a prototype. So if CTA chooses something else, probably we will follow
  suite and do one more backward-incompatible change at some point to align with CTA.


Testing of plotting functions
-----------------------------

Many of the data classes in Gammapy implement ``.plot()`` or ``.peek()`` methods to
allow users a quick look in the data. Those methods should be tested using the
`mpl_check_plot()` context manager. The context manager will take care of creating
a new figure to plot on and writing the plot to a byte-stream to trigger the
rendering of the plot, which can rasie errore as well. Here is a short example:

.. code-block:: python

    from gammapy.utils.testing import mpl_plot_check

    def test_plot():
        with mpl_plot_check():
            plt.plot([1., 2., 3., 4., 5.])


With this approach we make sure that the plotting code is at least executed once
and runs completely (up to saving the plot to file) without errors. In future we
will maybe change to something like https://github.com/matplotlib/pytest-mpl
to ensure that correct plots are produced.
