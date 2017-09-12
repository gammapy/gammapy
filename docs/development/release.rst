.. include:: ../references.txt

.. _development-release:

*****************************
How to make a Gammapy release
*****************************

This page contains step-by-step instructions how to make a Gammapy release.

Overview
--------

We have structured the procedure in three phases:

#. "Pre release" -- on a day (or several days) before making the release
#. "Make release"  -- on the day of making the release
   (tag the stable version and make source release on PyPI)
#. "Post release" -- on the day when announcing the release
   (ideally only two or three days after making the release)

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

Steps to prepare for the release (e.g. a week before) to check that things are in order:

#. Check the issue (example: https://github.com/gammapy/gammapy/issues/302 )
   and milestone (example: https://github.com/gammapy/gammapy/milestones/0.3 )
   for the release.
   Try to get developers to finish up their PRs, try to help fix bugs,
   and postpone non-critical issues to the next release.
#. Follow the instructions `here <http://docs.astropy.org/en/latest/development/affiliated-packages.html#updating-to-the-latest-template-files>`__
   to check that the astropy-helpers sub-module in Gammapy is pointing to the latest stable astropy-helpers release
   and whether there have been any fixes / changes to the Astropy
   `package-template <https://github.com/astropy/package-template/blob/master/TEMPLATE_CHANGES.md>`__
   since the last Gammapy release that should be copied over.
   In Gammapy we are using the method that's described in the section "managing the template files manually"
   that's described.
   If there are any updates to be done, you should do them via a PR so that travis-ci testing can run.
#. Do these extra checks and clean up any warnings / errors that come up::

       make trailing-spaces
       make code-analysis
       python setup.py test -V --remote-data

#. Check external HTML links from the docs (see :ref:`here <development-check_html_links>`).
#. Check that the travis-ci and readthedocs build is working (it usually always is).

   Links are at https://github.com/gammapy/gammapy#status-shields
#. Check that the changelog is complete, by going through the list of Github issues for the
   release milestone.

Make release
------------

Steps for the day of the release:

#. Mention release in the :ref:`gammapy_news` section.
#. Follow the instructions how to release an Astropy affiliated package
   `here <http://docs.astropy.org/en/latest/development/affiliated-packages.html#releasing-an-affiliated-package>`__.
#. Check that the tarball and description (which is from ``LONG_DESCRIPTION.rst``) on PyPI is OK.
#. Update the Gammapy stable branch to point to the new tag
   as described `here <http://docs.astropy.org/en/latest/development/releasing.html>`__.
#. Add the new version on readthedocs: https://readthedocs.org/dashboard/gammapy/versions/
   This will automatically trigger a build for that version, which you can check here:
   https://readthedocs.org/projects/gammapy/builds/
#. Draft the release announcement as a new file in https://github.com/gammapy/gammapy/tree/master/dev/notes
   (usually by copy & pasting the announcement from the last release)
#. Update the Gammapy conda-forge package at https://github.com/conda-forge/gammapy-feedstock
#. Update Gammapy Macports package at https://github.com/macports/macports-ports
#. Encourage the Gammapy developers to try out the new stable version (update and run tests)
   via the Github issue for the release and wait a day or two for feedback.

Post release
------------

Steps for the day to announce the release:

#. Send release announcement to the Gammapy mailing list and on Gammapy Slack
   (using the version you drafted in https://github.com/gammapy/gammapy/tree/master/dev/notes ).
#. If it's a big release with important new features or fixes,
   also send the release announcement to the following mailing lists
   (decide on a case by case basis, if it's relevant to the group of people):

    * https://groups.google.com/forum/#!forum/astropy-dev
    * https://lists.nasa.gov/mailman/listinfo/open-gamma-ray-astro
    * CTA DATA list (cta-wp-dm@cta-observatory.org)
    * hess-analysis
#. Make sure the release milestone and issue is closed on Github
#. Update these release notes with any useful infos / steps that you learned
   while making the release (ideally try to script / automate the task or check,
   e.g. as a ``make release-check-xyz`` target.
#. Open a milestone and issue for the next release (and possibly also a milestone for the
   release after, so that low-priority issues can already be moved there)
   Find a release manager for the next release, assign the release issue to her / him,
   and ideally put a tentative date (to help developers plan their time for the coming
   weeks and months).
#. Start working on the next release. :-)
