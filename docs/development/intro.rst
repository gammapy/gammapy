.. include:: ../references.txt

.. _dev_intro:

=============================
How to contribute to Gammapy?
=============================

What is this?
=============

**This page is an overview of how to make a change or addition to the Gammapy
code, tests or documentation. It's partly an introduction to the process, partly
a guide to some technical aspects.**

It is *not* a tutorial introduction explaining all the tools (git, Github,
Sphinx, pytest) or code (Python, Numpy, Scipy, Astropy) in detail. In the
Gammapy docs, we don't have such a tutorial introduction written up, but we're
happy to point out other tutorials or help you get started at any skill level
anytime if you ask.

Before attempting to make a contribution, you should *use* Gammapy a little bit
at least:

* Install Gammapy
* Execute one or two of the tutorial notebooks for Gammapy and do the exercises there.
* Ask questions or complain about issues on the Gammapy mailing list or issue tracker

We'd like to note that there are many ways to contribute to the Gammapy project.
For example if you mention it to a colleague or suggest it to a student, or if
you use it and acknowledge Gammapy in a presentation, poster or publication, or
if you report an issue on the mailing list, those are contributions we value.
The rest of this page though is concerned only with the process and technical
steps how to contribute a code or documentation change via a **pull request**
against the Gammapy repository.

So let's assume you've used Gammapy for a while, and now you'd like to fix or
add something to the Gammapy code, tests or docs. Here's the steps and commands
to do it ...

Get in touch early
==================

**Usually the first step, before doing any work, is to get in touch with the
Gammapy developers!**

Especially if you're new to the project, and don't have an overview of ongoing
activities, there's a risk that your work will be in vain if you don't talk to
us early. E.g. it could happen that someone else is currently working on similar
functionality, or that you've found a code or documentation bug and are willing
to fix it, but it then turns out that this was for some part of Gammapy that we
wanted to re-write or remove soon anyways.

Also, it's usually more fun if you get a *mentor* or *reviewer* early in the
process, so that you have someone to bug with questions and issues that come up
while executing the steps outlined below.

After you've done a few contributions to Gammapy and know about the status of
ongoing work, the best way to proceed is to file an issue or pull request on
Github at the stage where you want feedback or review. Sometimes you're not sure
how to best do something, and you start by discussing it on the mailing list or
in a Github issue. Sometimes you know how you'd like to do it, and you just code
or write it up and make a pull request when it's basically finished.

In any case, please keep the following point also in mind ...


Make small pull requests
========================

**Contributions to Gammapy happen via pull requests on Github. We like them small.**

So as we'll explain in mor detail below, the contribution cycle to Gammapy is roughly:

1. Get the latest development version (``master`` branch) of Gammapy
2. Make fixes, changes and additions locally
3. Make a pull request
4. Someone else reviews the pull request, you iterate, others can chime in
5. Someone else signs off on or merges your pull request
6. You update to the latest ``master`` branch

Then you're done, and can start using the new version, or start a new pull request
with further developments. It is possible and common to work on things in parallel
using git branches.

So how large should one pull request be?

Our experience in Gammapy (and others confirm, see e.g. `here
<https://alexgaynor.net/2015/dec/29/shrinking-code-review/>`__) is that smaller
is better. Working on a pull request for an hour or maximum a day, and having a
diff of a few to maximum a few 100 lines to review and discuss is pleasant.

A pull request that drags on for more than a few days, or that contains a diff
or 1000 lines, is almost always painful and inefficient for the person making
it, but even more so for the person reviewing it.

The worst case is if you start a bit pull request, put in a lot of hours, but
then don't have time to "finish" it, and it's sitting there for a week or a
month without getting merged. Then it's either blocking others that want to work
on the same part of the code or docs, or they do it, and then you have merge
conflicts to resolve when you come back to it. And coming back to a large pull
request after a long time always means a large investment of time for the
reviewer, because they probably have to re-read the previous discussion, and
look through the large diff again.

So pull requests that are small, e.g. one bug fix with the addition of one
regression test, or one new function or class or file, or one documentation
example, and that get reviewed and merged quickly (ideally the same day,
certainly the same week), are best.

.. _dev_setup:

Get set up
==========

.. warning::

    The rest of this page isn't written yet. It's almost identical to
    https://cta-observatory.github.io/ctapipe/getting_started/index.html so for
    now, see there. Also, we shouldn't duplicate content from
    http://astropy.readthedocs.io/en/latest/#developer-documentation but link
    there instead.

The first steps are basically identical to
https://cta-observatory.github.io/ctapipe/getting_started/index.html (until step
4, excluding 5) and
http://astropy.readthedocs.io/en/latest/development/workflow/get_devel_version.html
(up to *Create your own private workspace*). The following is a quick summary of
commands to set up an environment for Gammapy development:

.. code-block:: bash

    # Fork the gammapy repository on GitHub, https://github.com/gammapy/gammapy
    cd code  # Go somewhere on your machine where you want to code
    git clone https://github.com/[your-github-username]/gammapy.git
    cd gammapy
    conda env create -f environment-dev.yml
    source activate gammapy-dev
    # for conda versions >=4.4.0 you may have to execute
    #'conda activate gammapy-dev' instead
    git remote add gammapy git@github.com:gammapy/gammapy.git
    git remote rename origin [your-user-name]

It is also common to stick with the name ``origin`` for your repository and to
use ``upstream`` for the respository you forked from. In any case, you can use
``$ git remote -v`` to list all your configured remotes.

When developing gammapy you never want to work on the ``master`` branch, but
always on a dedicated feature branch. To create and switch the branch you are
working on (see also :ref:`dev_working_example`):

.. code-block:: bash

    git branch [branch-name]
    git checkout [branch-name]

To *activate* your development version (branch) of Gammapy in your environment:

.. code-block:: bash

    pip install -e .

This build is necessary to compile the few Cython code (``*.pyx``). If you skip
this step, some imports depending on Cython code will fail. This is described in
more details here: :ref:`setup_cython`. If you want remove the generated files
run ``make clean``.

For the development it is also convenient to fork and set up the
:ref:`dev_gammapy-extra`, as well as declaring some environment variables:

.. code-block:: bash

    # Fork the gammapy-extra repository on GitHub, https://github.com/gammapy/gammapy-extra
    cd code
    git clone https://github.com/[your-github-username]/gammapy-extra.git
    export GAMMAPY_EXTRA=$PWD/gammapy-extra

`$GAMMAPY_EXTRA`` is mainly used for testing purposes, on the contrary the datasets
provided in ``gammapy.catalog`` and those used in the tutorial Jupyter notebooks
should be in a different path declared in a `$GAMMAPY_DATA` environment variable.
You can download these datasets with `gammapy download datasets` and then point
your `$GAMMAPY_DATA` to the local path you have chosen.

.. code-block:: bash

    # Download GAMMAPY_DATA
    cd code
    gammapy download datasets --out GAMMAPY_DATA
    export GAMMAPY_DATA=$PWD/GAMMAPY_DATA

* install dependencies
* git clone dev version
* run tests
* build docs
* explain make and setup.py

.. _dev_working_example:

Make a working example
======================

* Explain "documentation driven development" and "test driven development"

* make a branch
* test in ``examples``
* ``import IPython; IPython.embed`` trick


Integrate the code in Gammapy
=============================

* move functions / classes to Gammapy
* move tests to Gammapy
* check tests locally
* check docs locally

Contribute with Jupyter notebooks
=================================

* check tests with user tutorials environment: `gammapy jupyter test --tutor`
* strip the output cells and format code: `gammapy jupyter strip` `gammapy jupyter black`
* diff stripped notebooks: `git diff mynotbeook.pynb`

Make a pull request
===================

* make a pull request
* check diff on Github
* check tests on travis-ci

Code review
===========

tbd

Close the loop
==============

tbd
