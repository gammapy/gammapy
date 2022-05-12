.. include:: references.txt

.. image:: _static/gammapy_banner.png
    :width: 400px

|

Gammapy
-------
**Date**: |today| **Version**: |version|

**Useful links**:
`Web page <https://gammapy.org>`__ |
`Recipes <https://gammapy.github.io/gammapy-recipes>`__  |
`Discussions <https://github.com/gammapy/gammapy/discussions>`__ |
`Acknowledging <https://gammapy.org/acknowledging.html>`__ |
`Contact <https://gammapy.org/contact.html>`__


Gammapy is a community-developed, open-source Python package for gamma-ray
astronomy built on Numpy, Scipy and Astropy. **It is the core library for the** `CTA`_ **Science Tools**
but can also be used to analyse data from existing imaging atmospheric Cherenkov telescopes
(IACTs), such as `H.E.S.S.`_, `MAGIC`_ and `VERITAS`_. It also provides some support
for `Fermi-LAT`_ and `HAWC`_ data analysis.

Gammapy v0.20 is the release candidate for v1.0 and is considered feature complete.

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: _static/index_getting_started.svg

    Getting started
    ^^^^^^^^^^^^^^^

    New to *Gammapy*? Check out the getting started documents. They contain information
    on how to install and start using *Gammapy* on your local desktop computer.

    +++

    .. link-button:: getting-started/index
            :type: ref
            :text: To the quickstart docs
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index_user_guide.svg

    User guide
    ^^^^^^^^^^

    The user guide provide in-depth information on the
    key concepts of Gammapy with useful background information and explanation,
    as well as tutorials in the form of Jupyter notebooks.

    +++

    .. link-button:: userguide/index
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index_api.svg

    API reference
    ^^^^^^^^^^^^^

    The reference guide contains a detailed description of
    the Gammapy API. The reference describes how the methods work and which parameters can
    be used. It assumes that you have an understanding of the key concepts.

    +++

    .. link-button:: api-ref
            :type: ref
            :text: To the reference guide
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index_contribute.svg

    Developer guide
    ^^^^^^^^^^^^^^^

    Saw a typo in the documentation? Want to improve
    existing functionalities? The contributing guidelines will guide
    you through the process of improving Gammapy.

    +++

    .. link-button:: development/intro
            :type: ref
            :text: To the developer guide
            :classes: btn-block btn-secondary stretched-link


.. toctree::
    :maxdepth: 1
    :titlesonly:
    :hidden:

    getting-started/index
    userguide/index
    tutorials/index
    api.rst
    development/index
    changelog/index
