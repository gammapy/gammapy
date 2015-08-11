.. include:: ../references.txt

.. _development-release:

*****************************
How to make a Gammapy release
*****************************

This page contains step-by-step instructions how to make a Gammapy release.

We have structured the procedure in three phases:

#. "Pre release" -- before the release day
#. "Make release"  -- on the release day
#. "Post release" -- in the days after the release

The purpose of writing the procedure down explicitly is to make it easy for anyone to make a release
(as opposed to one or a few people with secret knowledge how to do it).
It's also good to not have to remember this stuff, and to avoid errors or forgotten steps.

Note that we currently don't do bugfix releases, i.e. making a release is always simple in the sense
that you just need to create a branch off of ``master``.
Making bugfix releases would be more difficult and involve identifying commits with bug fixes and
backporting those to the stable branches.

In these notes we'll use the Gammapy 0.3 release as an example.

Pre release
-----------

#. Make a milestone for the release.

   Example: https://github.com/gammapy/gammapy/milestones/0.3

   Go through the open issues and pull requests and make sure a milestone is set for all.
   Then set a realistic release date well in advance (ideally a month, although a week
   is probably OK) and follow up on these issues / pull requests.

   It's usually OK to just move issues and pull requests to the next milestone if
   they don't get done before the release date.

#. Make in issue for the release.

   Example: https://github.com/gammapy/gammapy/issues/302

   This can be used to take notes and discuss any release-related issues.

#. Do these extra checks and clean up any warnings / errors that come up::

       make code-analysis
       make trailing-spaces
       python setup.py test -V --remote-data

#. Check external HTML links from the docs (see :ref:`here <development-check_html_links>`).

#. Check that the travis-ci and readthedocs build is working (it usually always is).

   Links are at https://github.com/gammapy/gammapy#status-shields

Make release
------------

To make a Gammapy release, follow the instructions how to release an Astropy affiliated package
`here <http://astropy.readthedocs.org/en/latest/development/affiliated-packages.html#releasing-an-affiliated-package>`__.

Here's some additional notes / things to check:

#. Update the Gammapy version number on the :ref:`gammapy_welcome` section and the :ref:`install` section.
#. Mention release in the :ref:`gammapy_news` section.
#. After making the tag and release, update the Gammapy stable branch to point to the new tag
   as described `here <http://astropy.readthedocs.org/en/latest/development/releasing.html>`__.
#. Check that the tarball and description (which is from ``LONG_DESCRIPTION.rst``) on PyPI is OK.
#. Check that the new release shows up OK on readthedocs (TODO: need to add the new tag manually?)
#. Make a pull request that updates the Gammapy version number in this file to trigger a conda package build:
   https://github.com/astropy/conda-builder-affiliated/blob/master/requirements.txt

This step ends with the release tarball to PyPI and the tag on Github.

Post release
------------

Wait for a day (or a few days) until the new Gammapy conda package becomes available in the Astropy
conda channel at https://anaconda.org/astropy and until a few co-developers had the time to try out
the new release a bit.


#. Send an announcement to the Gammapy and Astropy mailing lists
   (TODO: we need a template for this, so that it's quick to do)

#. Update Debian package




