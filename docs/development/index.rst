.. _development:

***********
Development
***********

This page is a collection of notes for Gammapy developers and maintainers.

Note that Astropy has very extensive developer documentation
`here <http://astropy.readthedocs.org/en/latest/#developer-documentation>`__,
this page should only mention Gammapy-specific things.

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
spread across multiple modules (`.py` files), but the user shouldn't care about their
names, that would be too much to remember.

The only reason to import from a module directly is if you need to access a private
function, class or variable (something that is not listed in `__all__` and thus not
imported into the end-user namespace. 

Note that this means that in the definition of an "end-user namespace", e.g. in the
``gammapy/data/__init__.py`` file, the imports have to be sorted in a way such that
modules in ``gammapy/data`` are loaded when imported from other modules in that sub-package. 

Why we don't sub-class other data classes
_________________________________________

We have considered re-using data classes developed by others,
namely `~astropy.nddata.NDData` and the
`spectral_cube.SpectralCube <http://spectral-cube.readthedocs.org/en/latest/index.html>`__
classes.

But in both cases, the data model didn't really fit our use cases for gamma-ray astronomy
and so we decided to write our own data classes from scratch.

Here's some issues where this was discussed:
* https://github.com/radio-astro-tools/spectral-cube/issues/110
* https://github.com/astropy/astropy/pull/2855#issuecomment-52610106


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

Wipe readthedocs
----------------

As described `here <http://read-the-docs.readthedocs.org/en/latest/builds.html#deleting-a-stale-or-broken-build-environment>`__,
if the docs on readthedocs show old stuff, you need to first log in `here <https://readthedocs.org/accounts/login/>`__
and then wipe it to create a fresh / clean version by hitting `this URL <http://readthedocs.org/wipe/gammapy/latest/>`_
and then clicking the "wipe version" button.

You don't get a confirmation that the wipe has taken place, but you can check
`here <https://readthedocs.org/builds/gammapy/>`__ (wait a few minutes)
and if needed manually start a new build by going
`here <https://readthedocs.org/projects/gammapy/>`__ and clicking the "Build" button.

Skip unit tests for some Astropy versions
-----------------------------------------

.. code-block:: python

   import astropy
   from astropy.tests.helper import pytest

   ASTROPY_VERSION = (astropy.version.major, astropy.version.minor)
   @pytest.mark.xfail(ASTROPY_VERSION < (0, 4), reason="Astropy API change")
   def test_something():
      ...

Make a Gammapy release
----------------------

For now, see https://github.com/astropy/package-template/issues/103
