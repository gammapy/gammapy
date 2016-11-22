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

#. Follow the instructions `here <http://docs.astropy.org/en/latest/development/affiliated-packages.html#updating-to-the-latest-template-files>`__
   to check that the astropy-helpers sub-module in Gammapy is pointing to the latest stable astropy-helpers release
   and whether there have been any fixes / changes to the Astropy
   `package-template <https://github.com/astropy/package-template/blob/master/TEMPLATE_CHANGES.md>`__
   since the last Gammapy release that should be copied over.
   In Gammapy we are using the method that's described in the section "managing the template files manually"
   that's described.
   If there are any updates to be done, you should do them via a PR so that travis-ci testing can run.

#. Do these extra checks and clean up any warnings / errors that come up::

       make code-analysis
       make trailing-spaces
       python setup.py test -V --remote-data

#. Check external HTML links from the docs (see :ref:`here <development-check_html_links>`).

#. Check that the travis-ci and readthedocs build is working (it usually always is).

   Links are at https://github.com/gammapy/gammapy#status-shields

#. Check that the changelog is complete, by going through the list of Github issues for the
   release milestone.

Make release
------------

These are the steps you should do on the day of the release:

#. Mention release in the :ref:`gammapy_news` section.
#. Follow the instructions how to release an Astropy affiliated package
   `here <http://docs.astropy.org/en/latest/development/affiliated-packages.html#releasing-an-affiliated-package>`__.
#. Check that the tarball and description (which is from ``LONG_DESCRIPTION.rst``) on PyPI is OK.
#. Update the Gammapy stable branch to point to the new tag
   as described `here <http://docs.astropy.org/en/latest/development/releasing.html>`__.
#. Add the new version on readthedocs: https://readthedocs.org/dashboard/gammapy/versions/
   This will automatically trigger a build for that version, which you can check here:
   https://readthedocs.org/projects/gammapy/builds/
#. Update the Gammapy conda package by updating these files:
     - https://github.com/astropy/conda-channel-astropy/blob/master/requirements.yml
     - https://github.com/astropy/conda-channel-astropy/blob/master/recipe_templates/gammapy/meta.yaml

Post release
------------

Wait for a day (or a few days) until the new Gammapy conda package becomes available in the Astropy
conda channel at https://anaconda.org/astropy and until a few co-developers had the time to try out
the new release a bit.


#. Send an announcement to the Gammapy and Astropy mailing lists
   (TODO: we need a template for this, so that it's quick to do)

#. Update Debian package




