.. include:: ../references.txt

.. _dev_things:

=====================
Gammapy project setup
=====================

This page gives an overview of the technical infrastructure we have set up to
develop and maintain Gammapy.

If you just want to make contribution to the Gammapy code or documentation, you
don't need to know about most of the things mentioned on this page.

But for Gammapy maintainers it's helpful to have a reference that explains what
we have and how things work.

gammapy repository
==================

This section explains the content of the main repository for Gammapy:

    https://github.com/gammapy/gammapy

Package and docs
----------------

The two main folders of interest for developers are the ``gammapy`` folder and
the ``docs`` folder. In ``gammapy`` you find the Gammapy package, i.e. all code,
but also tests are included there in sub-folders called ``tests``. The ``docs``
folder contains the documentation pages mostly in restructured text (RST) format. The
Sphinx documentation generator is used to convert those RST files to the HTML
documentation.

Notebooks
---------

The ``docs/tutorials`` folder contains tutorials that are part of the user
documentation for Gammapy in the form of Jupyter notebooks. There may be also notebooks in
other parts of the documentation. Except those specifically declared (see :ref:`skip-nb-execution`), all
the notebooks present in the documentation are executed during the doc building process. They are all
converted to the Sphinx-formatted HTML files, where clean output stripped ``.ipynb`` notebooks files
and ``.py`` scripts versions are generated and placed in the ``docs/_static/notebooks`` folder.

We perform automated set-up testing for all notebooks (just check that they run
and don't raise an exception) during the CI process (see below). It is also possible to
perform tests locally on notebooks  with the ``gammapy jupyter`` command. This
command provides functionalities for testing, code formatting, stripping output cells
and execution. See ``gammapy jupyter -h`` for more info on this.

The ``gammapy download`` command allows to download notebooks published in the documentation
as well as the related datasets needed to execute them. The set of notebooks is versioned
for each stable release as tar bundles published within the versioned documentation in the
`gammapy-docs <https://github.com/gammapy/gammapy-docs>`__ repository.
The same happens for conda working environments of stable releases, whose yaml files are published
in the `gammapy-web <https://github.com/gammapy/gammapy-webpage>`__ repository. The datasets are not
versioned and they are placed in the `gammapy-data <https://github.com/gammapy/gammapy-data>`__
repository.

.. _dev_build:

Build
-----

The ``setup.py`` and ``Makefile`` contain code to build and install Gammapy, as
well as to run the tests and build the documentation, see :ref:`dev_intro`.

The ``environment-dev.yml`` file contains the conda environment specification
that allows one to quickly set up a conda environment for Gammapy development,
see :ref:`dev_setup`.

The ``astropy_helpers`` folder is a git submodule used within ``setup.py`` (also
using ``ah_bootstrap.py``) which provides helpers related to Python build,
installation and packaging, including a robust way to build C and Cython code
from ``setup.py``, as well as pytest extensions for testing and Sphinx
extensions for the documentation build. If you look into those Python files, you
will find that they are highly complex, and full of workarounds for old versions
of Python, setuptools, Sphinx etc. Note that this is not code that we develop
and maintain in Gammapy. Gammapy was started from the
`astropy package template <https://github.com/astropy/package-template>`__ and there
are besides the ``astropy_helpers`` folder a few files (``ah_bootstrap.py``, ``setup.py``
``setup.cfg`` and ``gammapy/_astropy_init.py``) that are needed, but rarely
need to be looked at or updated. The Astropy team has set up a bot that from time to time makes pull
requests to update the affiliated packages (including Gammapy) as new versions
of ``astropy_helpers`` and the extra files are released.

Version
-------

One more thing worth pointing out is how versioning for Gammapy works. Getting a
correct version number in all cases (stable or dev version, installed package or
in-place build in the source folder, ...) is surprisingly complex for Python
packages. For Gammapy, the version is computed at build time, by ``setup.py``
calling into the ``get_git_devstr`` helper function, and writing it to the
auto-generated file ``gammapy/version.py``. This file is then part of the
Gammapy package, and is imported via ``gammapy/_astropy_init.py`` from
``gammapy/__init__.py``. This means that one can simply do this and always get
the right version for Gammapy::

    import gammapy
    gammapy.__version__
    gammapy.__githash__

.. _setup_cython:

Cython
------

We also have some Cython code in Gammapy, at the time of this writing less than
1% in this file:

* ``gammapy/stats/fit_statistics_cython.pyx``

and again as part of the Astropy package template there is the
``gammapy/_compiler.c`` file to help ``setup.py`` figure out information about
the C compiler at build time. These are the files that are compiled by Cython
and your C compiler when you build the Gammapy package, as explained in
:ref:`dev_intro`.

Other
-----

There are two more folders in the ``gammapy`` repository: ``examples`` and ``dev``.
We started with the ``examples`` folder with the idea to have Gammapy usage
examples there and have them be part of the user documentation. But this is not
the case at the moment, rather ``examples`` is a collection of scripts that have
mostly been used by developers to develop and debug Gammapy code. Most can
probably just be deleted, some should be moved to user documentation (not clear
where, could move all content to notebooks) or automated tests. The idea for the
``dev`` folder was to just have a place for scripts and checks and notes by
Gammapy developers. Like for ``examples``, it's mostly outdated cruft and should
probably be cleaned out.

The file in ``github/workflows/ci.yml`` is the configuration file for the continuous
integration (CI) we use with Github actions.

Finally, there are some folders that are generated and filled by various build
steps:

* ``build`` contains the Gammapy package if you run ``python setup.py build``.
  If you run ``python setup.py install``, first the build is run and files
  placed there, and after that files are copied from the ``build`` folder to
  your ``site-packages``.
* ``docs/_build`` contains the generated documentation, especially ``docs/_build/html`` the HTML version.
* ``htmlcov`` and ``.coverage`` is where the test coverage report is stored.
* ``v`` is a folder Pytest uses for caching information about failing tests across test runs.
  This is what makes it possible to execute tests e.g. with the ``--lf`` option
  and just run the tests that "last failed".
* ``dist`` contains the Gammapy distribution if you run ``python setup.py sdist``

.. _dev_gammapy-extra:

gammapy-extra repository
========================

For Gammapy we have a second repository for most of the example data files and
a few other things:

    https://github.com/gammapy/gammapy-extra

Old example data
----------------

The ``datasets`` and ``datasets/tests`` folders contain example datasets that were
used by the Gammapy documentation and tests. Note that here is a lot of old cruft,
because Gammapy was developed since 2013 in parallel with the development of data
formats for gamma-ray astronomy (see below).

Many old files in those folders can just be deleted; in some cases where
documentation or tests access the old files, they should be changed to access
newer files or generate test datasets from scratch. Doing this "cleanup" and
improvement of curated example datasets will be an ongoing task in Gammapy for
the coming years, that has to proceed in parallel with code, test and
documentation improvements.

The datasets needed were moved to the `gammapy-data <https://github.com/gammapy/gammapy-data>`__
repository, which is the dataset repository that is used now to perform the CI tests, build
the documentation and check notebooks and tutorials.

Other
-----

* The ``figures`` folder contains images that we show in the documentation (or
  in presentations or publications), for cases where the analysis and image
  takes a while to compute (i.e. something we don't want to do all the time
  during the Gammapy documentation build). In each case, there should be a
  Python script to generate the image.
* The ``experiments`` and ``checks`` folders contain Python scripts and
  notebooks with, well, experiments and checks by Gammapy developers. Some are
  still work in progress and of interest, most could probably be deleted.
* The ``logo`` folder contains the Gammapy logo and banner in a few different variants.
* The ``posters`` and ``presentations`` folders contain a few Gammapy posters
  and presentations, for cases where the poster or presentation isn't available
  somewhere else on the web. It's hugely incomplete and probably not very useful
  as-is, and we should discuss if this is useful at all, and if yes, how we want
  to maintain it.

Other repositories
==================

Datasets repository:

* https://github.com/gammapy/gammapy-data

Performance benchmarks for Gammapy:

* https://github.com/gammapy/gammapy-benchmarks

Data from tutorials sometimes accesses files here:

* https://github.com/gammapy/gamma-cat
* https://github.com/gammapy/gammapy-fermi-lat-data

Information from meetings is here:

* https://github.com/gammapy/gammapy-meetings

Gammapy webpages
================

There are two webpages for Gammapy: http://gammapy.org and http://docs.gammapy.org.

In addition we have Binder set up to allow users to try Gammapy in the browser.

gammapy.org
-----------

https://gammapy.org/ is a small landing page for the Gammapy project. The page
shown there a static webpage served via Github pages.

To update it, edit the HTML and CSS files in the `gammapy-webpage`_ repo
and then make a pull request against
the default branch for that repo, called ``gh-pages``. Once it's merged, the
webpage at https://gammapy.org/ usually updates within less than a minute.

docs.gammapy.org
----------------

https://docs.gammapy.org/ contains most of the documentation for Gammapy,
including information about Gammapy, the changelog, tutorials,...

The dev version of the docs may be built and updated with a manual github action. All
the docs are versioned, and each version of the docs is placed in its dedicated
version-labelled folder. It is recommended to build the docs locally before each release
to identify and fix possible Sphinx warnings from badly formatted RST files or failing
Python scripts used to display figures.

Gammapy Binder
--------------

We have set up https://mybinder.org/ for Gammapy, which allows users to execute
the notebooks present in the docs within the web browser, without having to install
software or download data to their local machine. This can be useful for people
to get started, and for tutorials. Every HTML-fixed version of the notebooks
that you can find in the :ref:`tutorials` section has a link to Binder that allows
you to execute the tutorial in the myBinder cloud infrastructure.

myBinder provides versioned virtual environments coupled with every release and
the master branch of the `gammapy Github repository <https://github.com/gammapy/gammapy>`__ .
The myBinder docker image is created using the ``Dockerfile`` and ``binder.py`` files placed
in the master branch of the `gammapy-webpage Github repository <https://github.com/gammapy/gammapy-webpage>`__.
The Dockerfile makes the Docker image used by Binder running some linux commands to install base-packages
and copy the notebooks and datasets needed. It executes ``binder.py`` to conda
install Gammapy dependencies listed in the environment YAML published within the versioned
documentation.

Continuous integration
======================

We are running various builds as Github actions workflows for CI.

Code quality
============

* Code coverage: https://coveralls.io/r/gammapy/gammapy
* Code quality: https://lgtm.com/projects/g/gammapy/gammapy/context:python
* Codacy: https://app.codacy.com/gh/gammapy/gammapy/dashboard

To run all tests and measure coverage, type the command ``make test-cov``::

    $ make test-cov

Releases
========

At this time, making a Gammapy release is a sequence of steps to execute in the
command line and on some webpages, that is fully documented in this checklist:
:ref:`dev-release`. It is difficult to automate this procedure more, but it is
already pretty straightforward and quick to do. If all goes well, making a
release takes about 1 hour of human time and one or two days of real time, with
the building of the conda binary packages being the slowest step, something we
wait for before announcing a new release to users (because many use conda and
will try to update as soon as they get the announcement email).

* Source distribution releases: https://pypi.org/project/gammapy/
* Binary conda packages for Linux, Mac and Windows:
  https://github.com/conda-forge/gammapy-feedstock
