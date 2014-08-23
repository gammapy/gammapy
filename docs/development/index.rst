.. _development:

***********
Development
***********

This page is a collection of notes for Gammapy developers and maintainers.

Note that Astropy has very extensive developer documentation
`here <http://astropy.readthedocs.org/en/latest/#developer-documentation>`__,
this page should only mention Gammapy-specific things.


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
