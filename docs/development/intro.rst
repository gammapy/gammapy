.. include:: ../references.txt

.. _dev_intro:

============================
How to contribute to Gammapy
============================

What is this?
=============

**This page is an overview of how to make a change or addition to the Gammapy
code, tests or documentation. It's partly an introduction to the process, partly
a guide to some technical aspects.**

It is *not* a tutorial introduction explaining all the tools (git, GitHub,
Sphinx, pytest) or code (Python, Numpy, Scipy, Astropy) in detail. In the
Gammapy docs, we don't have such a tutorial introduction written up, but we're
happy to point out other tutorials or help you get started at any skill level
anytime if you ask.

Before attempting to make a contribution, you should *use* Gammapy a bit at least:

* Install Gammapy.
* Execute one or two of the tutorial notebooks for Gammapy and do the exercises there.
* Ask questions or complain about issues on the Gammapy `slack channel <https://gammapy.slack.com/>`__, `GitHub discussions <https://github.com/gammapy/gammapy/discussions>`__ or `GitHub issues tracker <https://github.com/gammapy/gammapy/issues>`__.

We'd like to note that there are many ways to contribute to the Gammapy project.
For example if you mention it to a colleague or suggest it to a student, or if
you use it and `acknowledge Gammapy <https://gammapy.org/acknowledging.html>`__
in a presentation, poster or publication, or if you report an issue on the mailing list,
those are contributions we value. The rest of this page though is concerned only with
the process and technical steps how to contribute a code or documentation change via a
**pull request** against the Gammapy repository.

So let's assume you've used Gammapy for a while, and now you'd like to fix or
add something to the Gammapy code, tests or docs. Here are the steps and commands
to do it ...

Acceptation of the Developer Certificate of Origin (DCO)
========================================================
As described in the `PIG 24 <https://github.com/gammapy/gammapy/blob/master/docs/development/pigs/pig-024.rst>`_ and
the `README.rst <https://github.com/gammapy/gammapy/blob/master/README.rst>`_ file, each contributor shall accept the
DCO based stored in the file `CONTRIBUTING.md <https://github.com/gammapy/gammapy/blob/main/CONTRIBUTING.md#gammapy-developer-certification-of-origin>`_:
this is a binding statement that asserts that you are the creator of your contribution, and that you wish to allow
Gammapy to use your work to cite you as contributor.

**If you are willing to agree to these terms, the following agreement line should be added to every commit message:**

``Signed-off-by: Random J Developer <random@developer.example.org>``

Four solutions exist:

1. You add this message by hand into each of your commit messages (not recommended)

2. You can sign each of your commits with the command: "``git commit -s``".

If you have authored a commit that is missing its ‘Signed-off-by’ line, you can amend your commits and push them to
GitHub: "``git commit --amend --no-edit --signoff``"
(see also this `How To <https://github.com/src-d/guide/blob/master/developer-community/fix-DCO.md#how-to-add-sign-offs-retroactively>`_).

3. You can make an alias of the command "``git commit -s``", e.g.

``alias gcs 'git commit -s'``

4. You can create a so-called `git hooks` allowing to automatically sign all your commits (recommended option). This
method is described in detail `here <https://github.com/src-d/guide/blob/master/developer-community/fix-DCO.md#how-to-prevent-missing-sign-offs-in-the-future>`_.

For each of these solutions, it is **mandatory** to correctly set your `user.name` and `user.email` as part of your git
configuration (see `this page <https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-email-preferences/setting-your-commit-email-address>`_ to configure it).
You have to use **your real name** (i.e., pseudonyms or anonymous contributions cannot be made) when using git. This is
because the DCO is a binding document, granting the Gammapy project to be an open source project.


Get in touch early
==================

**Usually the first step, before doing any work, is to get in touch with the
Gammapy developers!**

Especially if you're new to the project, and don't have an overview of ongoing
activities, there's a risk that your work will be in vain if you don't talk to
us early. E.g. it could happen that someone else is currently working on similar
functionality, or that you've found a code or documentation bug and are willing
to fix it, but it then turns out that this was for some part of Gammapy that we
wanted to re-write or remove soon anyway.

Also, it's usually more fun if you get a *mentor* or *reviewer* early in the
process, so that you have someone to bug with questions and issues that come up
while executing the steps outlined below.

After you've done a few contributions to Gammapy and know about the status of
ongoing work, the best way to proceed is to file an issue or pull request on
GitHub at the stage where you want feedback or review. Sometimes you're not sure
how to best do something, and you start by discussing it on the mailing list or
in a GitHub issue. Sometimes you know how you'd like to do it, and you just code
or write it up and make a pull request when it's basically finished.

In any case, please keep the following point also in mind ...


Make small pull requests
========================

**Contributions to Gammapy happen via pull requests on GitHub. We like them small.**

So as we'll explain in more detail below, the contribution cycle to Gammapy is roughly:

1. Get the latest development version (``main`` branch) of Gammapy
2. Make fixes, changes and additions locally
3. Make a pull request
4. Someone else reviews the pull request, you iterate, others can chime in
5. Someone else signs off on or merges your pull request
6. You update to the latest ``main`` branch

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

The worst case is if you start a pull request, put in a lot of hours, but
then don't have time to "finish" it, and it's sitting there for a week or a
month without getting merged. Then it's either blocking others that want to work
on the same part of the code or docs, or they do it, and then you have merged
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

This section is based on the `instructions <https://ctapipe.readthedocs.io/en/latest/>`_
for setting up ``ctapipe``, a pipeline for the low-level processing of CTAO data.

Firstly, fork the `gammapy repository <https://github.com/gammapy/gammapy>`_ to your
personal GitHub. If you have not already done so, copy your public SSH key (normally under
``$HOME/.ssh/*.pub``) to your GitHub profile. This is done by going into your GitHub settings,
selecting **SSH and GPG keys** then **New SSH key** and then choosing **Authentication Key**
as the key type. If you do not have a SSH key, you may create it following the instructions on
`this page <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent>`_.

Go to the location on your machine where you want to clone the gammapy
repository from your GitHub:

.. code-block:: bash

    cd [your-working-path]
    git clone git@github.com:[your-github-username]/gammapy.git

Create the conda environment and activate it:

.. code-block:: bash

    cd gammapy
    conda env create -f environment-dev.yml
    conda activate gammapy-dev

Instead of Conda, you may use `Mamba <https://mamba.readthedocs.io/en/latest/>`_ as your
package manager, which offers higher installation speed and more reliable environment solutions.
In that case, use ``mamba env create -f environment-dev.yml`` to create the environment.

If you are already working with the development version environment and you want
to update it with the contents of ``environment-dev.yml`` run the following command:

.. code-block:: bash

    conda env update --file environment-dev.yml --prune

Add the gammapy repository as your upstream remote:

.. code-block:: bash

    git remote add upstream git@github.com:gammapy/gammapy.git

You can use ``git remote -v`` to list all your configured remotes.
If you followed the steps above, you should see something like this:

.. code-block:: bash

    origin	git@github.com:[your-github-username]/gammapy.git (fetch)
    origin	git@github.com:[your-github-username]/gammapy.git (push)
    upstream	git@github.com:gammapy/gammapy.git (fetch)
    upstream	git@github.com:gammapy/gammapy.git (push)

Make your ``main`` branch track the gammapy repository:

.. code-block:: bash

    git fetch upstream
    git branch --set-upstream-to=upstream/main

When developing Gammapy you never want to work on the ``main`` branch, but always
on a dedicated feature branch. You can create a new branch with:

.. code-block:: bash

    git branch [branch-name]

To switch between different branches:

.. code-block:: bash

    git checkout [branch-name]

You can also do both things at once as:

.. code-block:: bash

    git checkout -b [branch-name]

To *activate* your development version (branch) of Gammapy in your environment:

.. code-block:: bash

    python -m pip install -e .

This is necessary to compile the few Cython code  (``*.pyx``). If you skip
this step, some imports depending on Cython code will fail. If you want to remove the generated
files run ``make clean``.

Every time that you want to ``push`` your local branch to your GitHub repository you should run:

.. code-block:: bash

    git push origin [branch-name]

Download the Gammapy datasets and set the corresponding environment variable inside
the conda environment:

.. code-block:: bash

    gammapy download datasets --out gammapy-datasets
    conda env config vars set GAMMAPY_DATA=$PWD/gammapy-datasets/dev
    conda activate gammapy-dev

Alternatively, you may include the line ``export GAMMAPY_DATA=$PWD/gammapy-datasets/dev``
to your ``$HOME/.bashrc`` or ``$HOME/.bash_profile``, so that the variable is available
in every terminal session, independent from the active conda environment.

We adhere to the PEP8 coding style. To enforce this, setup the
`pre-commit hook <https://pre-commit.com/>`_:

.. code-block:: bash

    pre-commit install


Running tests & building Documentation
======================================
To run tests and build documentation we use tool `tox <https://tox.wiki/en/latest/>`__.
It is a virtual environment management tool which allows you to test Gammapy locally
in mutltiple test environments with different versions of Python and our dependencies.
It is also used to build the documentation and check the codestyle in a specific environment.
The same setup based on `tox` is used in our CI build.

Once you have created and activated the `gammapy-dev` environment, made some modification
to the code, you should run the tests:

.. code-block:: bash

    tox -e test

This will execute the tests in the standard `test`` environment. If you would like
to test with a different environment you can use:

.. code-block:: bash

    tox -e py310-test-numpy121

Which will test the code with Python 3.10 and numpy 1.21. All available pre-defined
environments can be listed using:

.. code-block:: bash

    tox --listenvs

However for most contributions testing with the standard `tox -e test` command is sufficient.
Additional arguments for `pytest` can be passed after `--`:

.. code-block:: bash

    tox -e test -- -n auto

Of course you can always use `pytest <https://docs.pytest.org/en/7.1.x/>`__ directly to
run tests, e.g. to run tests in a specific sub-package:

.. code-block:: bash

    pytest gammapy/maps

To build the documentation locally you can use:

.. code-block:: bash

    tox -e build_docs

And use `make docs-show` to open a browser and preview the result.

The codestyle can be checked using the command:

.. code-block:: bash

    tox -e codestyle

Which will run the tool `flake8` to check for code style issues.


..
    * run tests
    * build docs
    * explain make and setup.py

    **Make a working example**

    * Explain "documentation driven development" and "test driven development"

    * make a branch
    * test in ``examples``
    * ``import IPython; IPython.embed`` trick

    **Integrate the code in Gammapy**

    * move functions / classes to Gammapy
    * move tests to Gammapy
    * check tests locally
    * check docs locally

    **Contribute with Jupyter notebooks**

    * check tests with user tutorials environment: `gammapy jupyter --src mynotebook.ipynb test --tutor`
    * strip the output cells: `gammapy jupyter --src mynotebook.ipynb strip`
    * clean format code cells: `gammapy jupyter --src mynotebook.ipynb  black`
    * diff stripped notebooks: `git diff mynotbeook.pynb`

    **Make a pull request**

    * make a pull request
    * check diff on GitHub
    * check tests on travis-ci

    **Code review**

    tbd

    **Close the loop**

    tbd
