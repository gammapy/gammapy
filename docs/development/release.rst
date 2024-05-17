.. include:: ../references.txt

.. _dev-release:

*****************************
How to make a Gammapy release
*****************************

This page contains step-by-step instructions how to make a Gammapy release. We mostly follow the
`Astropy release instructions <https://docs.astropy.org/en/latest/development/releasing.html>`__
and just lists the additional required steps.


Feature Freeze and Branching
----------------------------

#. Follow the `Astropy feature freeze and branching instructions <https://docs.astropy.org/en/latest/development/releasing.html#start-of-a-new-release-cycle-feature-freeze-and-branching>`__
   Instead of updating the ``whatsnew/<version>.rst`` update the ``docs/release-notes/<version>.rst``.
#. Update the entry for the feature freeze in the `Gammapy release calendar <https://github.com/gammapy/gammapy/wiki/Release-Calendar>`__.


Releasing the first major release candidate
-------------------------------------------

A few days before the planned release candidate:

#. Fill the changelog ``docs/release-notes/<version>.rst`` for the version you are about to release.
#. Update the author list manually in the  ``CITATION.cff``.
#. Open a PR including both changes and mark it with the ``backport-v<version>.x`` label.
   Gather feedback from the Gammapy user and dev community and finally merge and backport to the ``v<version>.x`` branch.

On the day of the release candidate:

#. Add an entry for the release candidate like ``v1.0rc1`` or ``v1.1rc1`` in the ``download/index.json`` file in the `gammapy-web repo <https://github.com/gammapy/gammapy-webpage>`__, by
   copying the entry for ``dev`` tag. As we do not handle release candidates nor bug fix releases for data, this still allows to fix bugs in the data during the release candidate testing.
#. Update the ``CITATION.cff`` date and version by running the ``dev/prepare-release.py`` script.
#. Locally create a new release candidate tag on the v1.0.x, like ``v1.0rc1`` for Gammapy and push. For details see the
   `Astropy release candidate instructions <https://docs.astropy.org/en/latest/development/releasing.html#tagging-the-first-release-candidate>`__.
#. Once the tag is pushed the docs build and upload to PyPi should be triggered automatically.
#. Once the docs build has succeded find the ``tutorials_jupyter.zip`` file for the release candidate
   in the `gammapy-docs repo <https://github.com/gammapy/gammapy-docs>`__ and adapt the ``download/index.json`` to point to it.
#. Update the entry for the release candidate in the `Gammapy release calendar <https://github.com/gammapy/gammapy/wiki/Release-Calendar>`__.
#. Create a testing page like `Gammapy v1.0rc testing <https://github.com/gammapy/gammapy/wiki/Gammapy-v1.0rc-testing>`__.
#. Advertise the release candidate and motivate developers and users to report test fails and bugs and list them
   on the page created before.


Releasing the final version of the major release
------------------------------------------------

#. Create a new release tag in the `gammapy-data repo <https://github.com/gammapy/gammapy-data>`__, like ``v1.0`` or ``v1.1``.

#. Update the datasets entry in the ``download/index.json`` file in the `gammapy-web repo <https://github.com/gammapy/gammapy-webpage>`__ to point
   to this new release tag.

#. Locally create a new release tag like ``v1.0`` for Gammapy and push. For details see the
   `Astropy release candidate instructions <https://docs.astropy.org/en/latest/development/releasing.html#tagging-the-first-release-candidate>`__,
   but leave out the ``rc1`` suffix.

#. In the `gammapy-docs repo <https://github.com/gammapy/gammapy-docs>`__:

   * Wait for the triggered docs build to finish.
   * Edit ``stable/switcher.json`` to add the new version.

#. In the `gammapy-web repo <https://github.com/gammapy/gammapy-webpage>`__:

   * Mention the release on the front page and on the news page.
   * In the ``download/install`` folder, copy a previous environment file file as ``gammapy-1.0-environment.yml``.
   * Adapt the dependency conda env name and versions as required in this file.
   * Adapt the entry in the ``download/index.json`` file to point ot the correct environment file.
   * Find the ``tutorials_jupyter.zip`` file for the new release in the `gammapy-docs repo <https://github.com/gammapy/gammapy-docs>`__
     and adapt the ``download/index.json`` to point to it.

#. Update the entry for the actual release in the `Gammapy release calendar <https://github.com/gammapy/gammapy/wiki/Release-Calendar>`__.

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
#. Update version number in Binder ``Dockerfile`` in
   `gammapy-webpage repository <https://github.com/gammapy/gammapy-webpage>`__ master branch
   and tag the release for Binder.
#. Open a milestone and issue for the next release (and possibly also a milestone for the
   release after, so that low-priority issues can already be moved there) Find a
   release manager for the next release, assign the release issue to her / him,
   and ideally put a tentative date (to help developers plan their time for the
   coming weeks and months).
#. Start working on the next release.


Make a Bugfix release
---------------------

#. Add an entry for the bug-fix release like ``v1.0.1`` or ``v1.1.2`` in the ``download/index.json`` file in the `gammapy-web repo <https://github.com/gammapy/gammapy-webpage>`__.
   The ``datasets`` entry should point to last stable version, like ``v1.0`` or ``v1.1``. We do not provide bug-fix release for data.

#. Follow the  `Astropy bug fix release instructions <https://docs.astropy.org/en/latest/development/releasing.html#maintaining-bug-fix-releases>`__.

#. Follow the instructions for a major release for the updates of CITATION.cff, the modifications in the `gammapy-docs` and `gammapy-webpage` repo as well as the conda builds.