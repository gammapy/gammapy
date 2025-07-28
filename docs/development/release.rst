.. include:: ../references.txt

.. _dev-release:

*****************************
How to make a Gammapy release
*****************************

This page contains step-by-step instructions for how to make a Gammapy release. The procedure shown here is inspired by
the `astropy release scheme <https://docs.astropy.org/en/latest/development/maintainers/releasing.html#release-procedure-for-the-astropy-core-package>`__.
The general procedure can be broken down into three major steps:


* Feature freeze: freeze the core package features and create a new branch for the release. This is typically done two
  weeks before the release.
* Release candidate: proposed feature release which should be tested. This is typically done one week before the
  final release.
* Final release.


Feature Freeze and Branching
----------------------------

**A few days before the feature freeze:**

#. Fill the changelog ``docs/release-notes/<version>.rst`` for the version you are about to release.

   * To generate the list of pull requests and issues run ``python dev/github_summary.py create_pull_request_table``.
     This will create a list of all closed pull requests and save it to ``table_pr.ecsv``.
     Next, use the option ``merged_PR`` to extract a relevant list corresponding to the release.
     You can then manually delete any entries which correspond to small improvements or bug fixes.
#. Update the author list manually in the  ``CITATION.cff``.

    * You can use the helper script ``dev/authors.py`` for this.
#. Open a PR including both changes and mark it with the ``backport-v<version>.x`` label.
   Gather feedback from the Gammapy user and dev community and finally merge and backport to the
   ``v<version>.x`` branch.

**On the day of the feature freeze:**

#. Add a new milestone to the `GitHub issue tracker <https://github.com/gammapy/gammapy/milestones>`__ for the version
   ``v<version>.x`` (this will also be used for the next bugfix release). Also create a ``backport-v<version>.x``
   `label <https://github.com/gammapy/gammapy/labels>`__.
#. Update your local ``main`` branch to the latest from remote::

    git fetch upstream --tags --prune
    git checkout -B main upstream/main

#. Create a new branch with the name of the version::

    git branch v<version>.x

#. Stay on the ``main`` branch and make a copy and update the ``docs/release-notes/<version>.rst``
#. Commit the changes and push to GitHub ``main``.
#. Update the entry for the feature freeze in the
   `Gammapy release calendar <https://github.com/gammapy/gammapy/wiki/Release-Calendar>`__.


Releasing the first major release candidate
-------------------------------------------

#. In the `gammapy-webpage repo <https://github.com/gammapy/gammapy-webpage>`__:

   * Add an entry for the release candidate like ``v1.0rc1`` or ``v1.1rc1`` in the ``download/index.json`` file,
     by copying the entry for ``dev`` tag. As we do not handle release candidates nor bug fix releases for data,
     this still allows to fix bugs in the data during the release candidate testing.
   * In the ``download/install`` folder, copy a previous environment file file as ``gammapy-1.0rc1-environment.yml``.
   * Adapt the dependency conda env name and versions as required in this file.

#. Switch to the correct branch and update the ``CITATION.cff`` date and version by running the
   ``dev/prepare-release.py`` script::

    git checkout v1.0.x
    python ./dev/prepare-release.py --release v1.0rc1

#. Commit and push the branch back to GitHub::

    git push upstream v1.0.x

#. Locally create a new release candidate tag on the ``v1.0.x``, like ``v1.0rc1`` for Gammapy and push::

    git tag -s v1.0rc1 -m "Tagging v1.0rc1"
    git push upstream v1.0.x

#. Once the tag is pushed, the ``release`` action in charge of packaging and uploading to `PyPi <https://pypi.org/>`__
   should be triggered automatically. Once complete, it will trigger the docs build on the  ``gammapy-docs``
   repository.
#. Check the ``Actions`` on `gammapy repo <https://github.com/gammapy/gammapy>`__  and 
   `gammapy-docs <https://github.com/gammapy/gammapy-docs>`__  to check that the necessary actions have started.
#. Once the docs build is successful find the ``tutorials_jupyter.zip`` file for the release candidate in the
   `gammapy-docs repo <https://github.com/gammapy/gammapy-docs>`__ and adapt the ``download/index.json`` to point to it.
#. Update the entry for the release candidate in the
   `Gammapy release calendar <https://github.com/gammapy/gammapy/wiki/Release-Calendar>`__.
#. Create a testing page like
   `Gammapy v1.0rc testing <https://github.com/gammapy/gammapy/wiki/Gammapy-v1.0rc-testing>`__.
#. Advertise the release candidate and motivate developers and users to report test fails and bugs and list them
   on the page created before.


Releasing the final version of the major release
------------------------------------------------

#. Create a new tag in the `gammapy-data repo <https://github.com/gammapy/gammapy-data>`__, like ``v1.0``
   or ``v1.1``.

#. In the `gammapy-webpage repo <https://github.com/gammapy/gammapy-webpage>`__:

   * In the ``download/install`` folder, copy a previous environment file as ``gammapy-1.0-environment.yml``.
   * Adapt the dependency conda env name and versions as required in this file. Make sure to move ``gammapy=1.0``
     into the dependencies list.
   * Update the datasets entry in the ``download/index.json`` to point to this new release tag. Also update the
     notebook entry, typically the link extensions are the same between versions.

#. Locally create a new release tag like ``v1.0`` for Gammapy and push::

    git tag -s v1.0 -m "Tagging v1.0"
    git push upstream v1.0.x

#. In the `gammapy-docs repo <https://github.com/gammapy/gammapy-docs>`__:

   * Kill the possible ``dev-docs`` build actions as they might interfere with the ``release`` docs build.
   * Wait for the triggered ``release`` docs build to finish.
   * Edit ``docs/stable/switcher.json`` to add the new version.

#. In the `gammapy-webpage repo <https://github.com/gammapy/gammapy-webpage>`__:

   * Find the ``tutorials_jupyter.zip`` file for the new release in the
     `gammapy-docs repo <https://github.com/gammapy/gammapy-docs>`__ and confirm the link in
     ``download/index.json`` is correct.
   * Mention the release on the front page and on the news page. Both the ``news.html`` and ``index.html`` should be
     edited at this step to include the correct version numbers.

#. Update the entry for the actual release in the
   `Gammapy release calendar <https://github.com/gammapy/gammapy/wiki/Release-Calendar>`__.

#. Finally:

   * The Gammapy conda-forge package at https://github.com/conda-forge/gammapy-feedstock should be automatically updated within hours and a PR opened. Check that this is the case and if not, perform the manual update of the recipe meta.yaml on your gammapy-feedstock fork and open the PR. Finally, when all tests for all distributions successfully ran, merge the PR.   
   * Encourage the Gammapy developers to try out the new stable version (update and run tests) via the GitHub
     issue for the release and wait a day or two for feedback.


Post release
------------

Steps for the day to announce the release:

#. Send release announcement to the Gammapy mailing list and on Gammapy Slack.
#. If it's a big release with important new features or fixes,
   also send the release announcement to the following mailing lists
   (decide on a case by case basis, if it's relevant to the group of people):

    * https://groups.google.com/forum/#!forum/astropy-dev
    * CTAO AS WG list (cta-wg-as@cta-observatory.org)
    * hess-forum list (hess-forum@lsw.uni-heidelberg.de)

#. Make sure the release milestone and issue is closed on GitHub.
#. Update these release notes with any useful infos / steps that you learned
   while making the release (ideally try to script / automate the task or check,
   e.g. as a ``make release-check-xyz`` target).
#. Update the version numbers in `gammapy-webpage repo <https://github.com/gammapy/gammapy-webpage>`__ ``master``
   branch to allow the Binder ``Dockerfile`` to be updated:

    * In `gammapy-webpage repo <https://github.com/gammapy/gammapy-webpage>`__ update the
      ``master`` branch::

        git checkout master
        git pull

    * Update the four files: ``postBuild``, ``requirements.txt``, ``runtime.txt``, and ``start`` in the ``master`` branch::

        git add postBuild requirements.txt runtime.txt start
        git commit -s -m "Update binder configuration for new release"
        git push origin master

    * Tag the new version and push to the upstream repository::

        git tag -s v1.3 -m "Tagging v1.3"
        git push origin v1.3

#. Open a milestone and issue for the next release (and possibly also a milestone for the
   release after, so that low-priority issues can already be moved there). Find a
   release manager for the next release, assign the release issue to them,
   and ideally put a tentative date (to help developers plan their time for the
   coming weeks and months).
#. Start working on the next release.


Make a Bugfix release
---------------------

#. Add an entry for the bug-fix release like ``v1.0.1`` or ``v1.1.2`` in the ``download/index.json`` file in the
   `gammapy-webpage repo <https://github.com/gammapy/gammapy-webpage>`__. The ``datasets`` entry should point to the
   last stable version, like ``v1.0`` or ``v1.1``. We do not provide bug-fix release for data.

#. Follow the `Astropy bug fix release instructions <https://docs.astropy.org/en/latest/development/maintainers/releasing.html#maintaining-bug-fix-releases>`__.

#. Follow the instructions for a major release for the updates of ``CITATION.cff``, the modifications in the
   `gammapy-docs` and `gammapy-webpage` repos as well as the conda builds.
