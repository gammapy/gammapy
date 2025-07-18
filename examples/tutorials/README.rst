.. include:: ../references.txt

.. _tutorials:

=========
Tutorials
=========

.. important::

    * It is **strongly** advised to first read :ref:`package_structure` of the User Guide before using the
      tutorials.

    * In general, all methods and classes are defined with default values that permit a good execution per default.
      In the tutorials, we frequently use extra values to just illustrate their usage.

    * The Gammapy library is used by many instruments and as consequence can not describe the specificities of each
      data release of each observatory. Get in touch with the observatory experts to get the best usage of a given
      data release.

This page lists the Gammapy tutorials that are available as `Jupyter`_ notebooks.
You can read them here, or execute them using a temporary cloud server in Binder.

To execute them locally, you have to first install Gammapy locally (see
:ref:`installation`) and download the tutorial notebooks and example datasets (see
:ref:`getting-started`). Once Gammapy is installed, remember that you can always
use ``gammapy tutorial setup`` to check your tutorial setup, or in your script with

.. code-block:: python

    from gammapy.utils.check import check_tutorials_setup
    check_tutorials_setup()

Gammapy is a Python package built on `Numpy`_ and `Astropy`_, so to use it
effectively, you have to learn the basics. Many good free resources are
available, e.g. `A Whirlwind tour of Python`_, the `Python data science
handbook`_ and the `Astropy Hands-On Tutorial`_.
