.. _development:

***********
Development
***********

This page is a collection of notes for Gammapy developers and maintainers.

Note that Astropy has very extensive developer documentation
`here <http://astropy.readthedocs.org/en/latest/#developer-documentation>`__,
this page should only mention Gammapy-specific things.

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

For now, see https://github.com/astropy/package-template/issues/103

* Check external HTML links (see :ref:`here <development-check_html_links>`).

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
* https://github.com/zblz/naima
* https://github.com/woodmd/gammatools
* https://github.com/kialio/VHEObserverTools

These are unofficial, unmaintained copies on open codes on Github:

* https://github.com/brefsdal/sherpa
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
