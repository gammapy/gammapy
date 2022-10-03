.. include:: ../references.txt

.. _dev-release:

*****************************
How to make a Gammapy release
*****************************

This page contains step-by-step instructions how to make a Gammapy release.

Overview
--------

We have structured the procedure in three phases:

#. "Pre release" -- on a day (or several days) before making the release
#. "Make release" -- on the day of making the release
   (tag the stable version and make source release on PyPI)
#. "Post release" -- on the day when announcing the release
   (ideally only two or three days after making the release)

The purpose of writing the procedure down explicitly is to make it easy for
anyone to make a release (as opposed to one or a few people with secret
knowledge how to do it). It's also good to not have to remember this stuff, and
to avoid errors or forgotten steps.

Note that we currently don't do bugfix releases, i.e. making a release is always simple in the sense
that you just need to create a branch off of ``master``.
Making bugfix releases would be more difficult and involve identifying commits with bug fixes and
backporting those to the stable branches.

In these notes we'll use the Gammapy 0.14 release as an example.

Pre release
-----------

Steps to prepare for the release (e.g. a week before) to check that things are in order:

#. Try to get developers to finish up their PRs, try to help fix bugs, and postpone non-critical issues to the next release.
#. Do these extra checks and clean up any warnings / errors that come up::

       make polish
       make pylint
       make flake8

#. Check external HTML links from the docs (see :ref:`here <dev-check_html_links>`).
#. Check that the GitHub actions CI build is working.

   Links are at https://github.com/gammapy/gammapy#status-shields
#. Check that the release notes are complete, by going through the list of GitHub issues for the
   release milestone.
#. Check the author list in CITATION.cff by running the ``dev/authors.py`` script. Manually
   fi any mismatches and finally sort by commits.


Make release
------------

Steps for the day of the release:

#. Follow the instructions how to release an Astropy affiliated package:

   * `<https://docs.astropy.org/en/stable/development/astropy-package-template.html>`__.

#. In the `gammapy-docs repo <https://github.com/gammapy/gammapy-docs>`__:

   * Follow the stable documentation procedure.
   * Edit ``stable/index.html`` to point to ``0.14/index.html``.
   * Edit ``stable/switcher.json`` to add the new version.

#. In the `gammapy-web repo <https://github.com/gammapy/gammapy-webpage>`__:

   * Mention the release on the front page and on the news page.
   * In the ``download/install`` folder, copy ``gammapy-0.13-environment.yml`` file as ``gammapy-0.14-environment.yml``.
   * Adapt the dependency conda env name and versions as required in this file.
   * Adapt the ``download/index.json`` file to include the new version.

#. Finally:

   * Update the Gammapy conda-forge package at https://github.com/conda-forge/gammapy-feedstock
   * Encourage the Gammapy developers to try out the new stable version (update and run tests) via the GitHub issue for the release and wait a day or two for feedback.


Post release
------------

Steps for the day to announce the release:

#. Send release announcement to the Gammapy mailing list and on Gammapy Slack.
#. If it's a big release with important new features or fixes,
   also send the release announcement to the following mailing lists
   (decide on a case by case basis, if it's relevant to the group of people):

    * https://groups.google.com/forum/#!forum/astropy-dev
    * https://lists.nasa.gov/mailman/listinfo/open-gamma-ray-astro
    * CTA DATA list (cta-wp-dm@cta-observatory.org)
    * hess-analysis
#. Make sure the release milestone and issue is closed on GitHub
#. Update these release notes with any useful infos / steps that you learned
   while making the release (ideally try to script / automate the task or check,
   e.g. as a ``make release-check-xyz`` target).
#. Update version number in Binder `Dockerfile` in
   `gammapy-webpage repository <https://github.com/gammapy/gammapy-webpage>`__ master branch
   and tag the release for Binder.
#. Open a milestone and issue for the next release (and possibly also a milestone for the
   release after, so that low-priority issues can already be moved there) Find a
   release manager for the next release, assign the release issue to her / him,
   and ideally put a tentative date (to help developers plan their time for the
   coming weeks and months).
#. Start working on the next release. :-)
