.. _development:

***********
Development
***********

This page is a collection of notes for Gammapy developers and maintainers.

Note that Astropy has very extensive developer documentation
`here <http://astropy.readthedocs.org/en/latest/#developer-documentation>`__,
this page should only mention Gammapy-specific things.

.. _make_clean:

How to clean up old files
-------------------------

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

We support Python 2.7 and 3.3 or later using a single code base.
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

.. _development-release_gammapy:

Make a Gammapy release
----------------------

To make a Gammapy release, follow the instructions how to release an Astropy affiliated package
`here <http://astropy.readthedocs.org/en/latest/development/affiliated-packages.html#releasing-an-affiliated-package>`__.

Here's some additional notes / things to check:

* Check external HTML links (see :ref:`here <development-check_html_links>`).
* Update the Gammapy version number on the :ref:`gammapy_welcome` section and the :ref:`install` section.
* Mention release in the :ref:`gammapy_news` section.
* After making the tag and release, update the Gammapy stable branch to point to the new tag
  as described `here <http://astropy.readthedocs.org/en/latest/development/releasing.html>`__.

After doing the release, check these things:

* Check that the tarball and description (which is from ``LONG_DESCRIPTION.rst``) on PyPI is OK.
* Check that the new release shows up OK on readthedocs.
* Check `here <https://github.com/gammapy/gammapy/tags>`__ that the release tag is present on Github
* Send announcement to Gammapy mailing list, Astropy mailing list and Twitter.

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

``from numpy.testing import assert_allclose``

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

``from astropy.tests.helper import assert_quantity_allclose``

at the top of the file and then just use ``assert_quantity_allclose``
for the tests.
