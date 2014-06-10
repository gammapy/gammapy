Dev
===

This directory is the development playground.

It contains e.g. scripts that use other packages to produce reference results for unit tests.

Notes
-----

Here's some notes for us to remember how to do some things:

* As described `here <http://read-the-docs.readthedocs.org/en/latest/builds.html#deleting-a-stale-or-broken-build-environment>`__,
  if the docs on readthedocs show old stuff, you need to first log in `here <https://readthedocs.org/accounts/login/>`__
  and then wipe it to create a fresh / clean version by hitting `this URL <http://readthedocs.org/wipe/gammapy/latest/>`_.

  You don't get a confirmation that the wipe has taken place, but you can check
  `here <https://readthedocs.org/builds/gammapy/>`__ (wait a few minutes)
  and if needed manually start a new build by going
  `here <https://readthedocs.org/projects/gammapy/>`__ and clicking the "Build" button.

* To skip a unit test depending on Astropy version number do this:

.. code-block:: python

   import astropy
   from astropy.tests.helper import pytest

   ASTROPY_VERSION = (astropy.version.major, astropy.version.minor)
   @pytest.mark.xfail(ASTROPY_VERSION < (0, 4), reason="Astropy API change")
   def test_something():
      ...
