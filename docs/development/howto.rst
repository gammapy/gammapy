.. include:: ../references.txt

.. _development_howto:

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

    $ rm -r build docs/_build docs/api htmlcov

These folders only contain generated files and are always safe to delete!
Most of the time you don't have to delete them, but if you e.g. remove or rename files or functions / classes,
then you should, because otherwise the old files will still be around and you might get confusing results,
such as Sphinx warnings or import errors or code that works locally because it uses old things, but fails
on travis-ci or for other developers.

* The ``build`` folder is where ``python setup.py build`` or ``python setup.py install`` generate files.
* The ``docs/api`` folder is where ``python setup.py build_sphinx`` generates [RST]_ files from the docstrings
  (temporary files part of the HTML documentation generation).
* The  ``docs/_build`` folder is where ``python setup.py build_sphinx`` generates the HTML and other Sphinx
  documentation output files.
* The ``htmlcov`` folder is where ``python setup.py test --coverage`` generates the HTML coverage report.

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

.. _development-import_from:

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

.. _development-data_subclasses:

Why we don't sub-class other data classes
-----------------------------------------

We have considered re-using data classes developed by others,
namely `~astropy.nddata.NDData` and the
`spectral_cube.SpectralCube <http://spectral-cube.readthedocs.org/en/latest/index.html>`__
classes.

But in both cases, the data model didn't really fit our use cases for gamma-ray astronomy
and so we decided to write our own data classes from scratch.

Here's some issues where this was discussed:

* https://github.com/radio-astro-tools/spectral-cube/issues/110
* https://github.com/astropy/astropy/pull/2855#issuecomment-52610106

.. _development-result_object:

Functions returning several values
----------------------------------

Functions that return more than a single value shouldn't return a list
or dictionary of values but rather a Python Bunch result object. A Bunch
is similar to a dict, except that it allows attribute access to the result
values. The approach is the same as e.g. the use of `~scipy.optimize.OptimizeResult`.
An example of how Bunches are used in gammapy is given by the `~gammapy.detect.TSMapResult`
class.

.. _development-python2and3:

Python 2 and 3 support
----------------------

We support Python 2.7 and 3.4 or later using a single code base.
This is the strategy adopted by most scientific Python projects and a good starting point to learn about it is
`here <http://python3porting.com/noconv.html>`__ and
`here <http://astropy.readthedocs.org/en/latest/development/codeguide.html#writing-portable-code-for-python-2-and-3>`__.

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

.. _development-wipe_readthedocs:

Wipe readthedocs
----------------

As described `here <http://read-the-docs.readthedocs.org/en/latest/builds.html#deleting-a-stale-or-broken-build-environment>`__,
if the docs on readthedocs show old stuff, you need to first log in `here <https://readthedocs.org/accounts/login/>`__
and then wipe it to create a fresh / clean version by hitting this URL::

   http://readthedocs.org/wipe/gammapy/latest/

and then clicking the "wipe version" button.

You don't get a confirmation that the wipe has taken place, but you can check
`here <https://readthedocs.org/builds/gammapy/>`__ (wait a few minutes)
and if needed manually start a new build by going
`here <https://readthedocs.org/projects/gammapy/>`__ and clicking the "Build" button.

.. _development-skip_tests:

Skip unit tests for some Astropy versions
-----------------------------------------

.. code-block:: python

   import astropy
   from astropy.tests.helper import pytest

   ASTROPY_VERSION = (astropy.version.major, astropy.version.minor)
   @pytest.mark.xfail(ASTROPY_VERSION < (0, 4), reason="Astropy API change")
   def test_something():
      ...

.. _development-check_html_links:


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

Check HTML links
----------------

There's two ways to check the docs for broken links.


This will check external links (not nice because you have to install first):

.. code-block:: bash

   $ python setup.py install
   $ cd docs; make linkcheck

To check all internal and external links use this `linkchecker <https://github.com/wummel/linkchecker>`__:

.. code-block:: bash

   $ pip install linkchecker
   $ linkchecker --check-extern docs/_build/html/index.html

Because Sphinx doesn't warn about some broken internal links for some reason,
we run ``linkchecker docs/_build/html/index.html`` on travis-ci,
but not with the ``--check-extern`` option as that would probably fail
randomly quite often whenever one of the external websites is down.


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
* https://github.com/cdeil/kapteyn-mirror

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
After all the
`FTOOLS <http://heasarc.gsfc.nasa.gov/ftools/ftools_menu.html>`__
always use ``clobber``.

The reason is that ``overwrite`` is clear to everyone, but ``clobber`` is defined by the dictionary
(e.g. see `here <http://dictionary.reference.com/browse/clobber>`__)
as "to batter severely; strike heavily. to defeat decisively. to denounce or criticize vigorously."
and isn't intuitively clear to new users.

Astropy uses both ``clobber`` and ``overwrite`` in various places at the moment.
For Gammapy we can re-visit this decision before the 1.0 release, but for now,
please be consistent and use ``overwrite``.

Pixel coordinate convention
---------------------------

All code in Gammapy should follow the Astropy pixel coordinate convention that the center of the first pixel
has pixel coordinates ``(0, 0)`` (and not ``(1, 1)`` as shown e.g. in ds9).
It's currently documented `here <http://photutils.readthedocs.org/en/latest/photutils/overview.html#coordinate-conventions>`__
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

.. _development_random:

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

This is just a recommendation, e.g. for `gammapy.data.SpectralCube.spectral_index`
we decided to use this shorter format, but for `gammapy.data.SpectralCube.flux` we
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
        return SkyCoord(lon, lat, unit='deg', frame='fk5')


Class attributes
++++++++++++++++

Class attributes (data members) and properties are currently a bit of a mess,
see `~gammapy.spectral.Spectralcube` as an example.
Attributes are listed in an *Attributes* section because I've listed them in a class-level
docstring attributes section as recommended `here`__.
Properties are listed in separate *Attributes summary* and *Attributes Documentation*
sections, which is confusing to users ("what's the difference between attributes and properties?").

.. __: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt#class-docstring

One solution is to always use properties, but that can get very verbose if we have to write
so many getters and setters. I don't have a solution for this yet ... for now I'll go read
`this`__ and meditate.

.. __: http://nbviewer.ipython.org/urls/gist.github.com/ChrisBeaumont/5758381/raw/descriptor_writeup.ipynb

TODO: make a decision on this and describe the issue / solution here.

Constructor parameters
++++++++++++++++++++++

TODO: should we put the constructor parameters in the class or ``__init__`` docstring?

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

Gammapy plotting style
----------------------

Figures and plots in the Gammapy docs use the same consistent plotting style,
that is defined in `gammapy.utils.mpl_style`.  The style is derived from the
astropy plotting style applying a few minor changes. Here are two examples:

	* :ref:`Crab MWL SED plot <crab-mwl-sed>`
	* :ref:`Fermi 1FHL skymap <fermi-1fhl-skymap>`

For the Gammapy docs the style is used by default and doesn't have to be set
explicitly. If you would like to use the style outside the Gammapy docs, add
the following lines to the beginning of your plotting script or notebook:

.. code-block:: python

	import matplotlib.pyplot as plt
	from gammapy.utils.mpl_style import gammapy_mpl_style
	plt.style.use(gammapy_mpl_style)
