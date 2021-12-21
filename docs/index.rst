.. include:: references.txt

.. Should we add a known issues section at the end?

gammapy documentation
---------------------

**Date**: |today| **Version**: |version|

**Useful links**:
:ref:`install` |
`Source Repository <https://github.com/gammapy/gammapy/issues>`__ |
`Issues & Ideas <https://github.com/gammapy/gammapy/issues>`__ |
`Discussions <https://github.com/gammapy/gammapy/discussions>`__ |
`Contact <https://gammapy.org/contact.html>`__


Gammapy is a community-developed, open-source Python package for gamma-ray
astronomy built on Numpy, Scipy and Astropy. It is the core library for the `CTA`_ science tools
and can also be used to analyse data from existing imaging atmospheric Cherenkov telescopes
(IACTs), such as `H.E.S.S.`_, `MAGIC`_ and `VERITAS`_. It also provides some support
for `Fermi-LAT`_ and `HAWC`_ data analysis.

This webpage contains the Gammapy documentation. You may also check out the `Gammapy webpage <https://gammapy.org>`_
where you may find more information about Gammapy, including the
`list of releases <https://gammapy.org/news.html#releases>`_ and contact information if you
have any questions, want to report and issue or request a feature, or need help with anything
related to Gammapy.

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    ---
    :img-top: _static/index_getting_started.svg

    Getting started
    ^^^^^^^^^^^^^^^

    New to *Gammapy*? Check out the getting started documents. They contain information
    on how to install and start using *Gammapy'* in your local desktop.

    +++

    .. link-button:: getting-started/index
            :type: ref
            :text: To the getting started docs
            :classes: btn-block btn-secondary stretched-link

    ---
    :img-top: _static/index_user_guide.svg

    Tutorials
    ^^^^^^^^^

    The tutorials provide in-depth information on the
    key concepts of Gammapy with useful background information and explanation.

    +++

    .. link-button:: tutorials/index
            :type: ref
            :text: To the tutorials
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

    .. link-button:: development/index
            :type: ref
            :text: To the developer guide
            :classes: btn-block btn-secondary stretched-link


.. toctree::
    :maxdepth: 1
    :titlesonly:
    :hidden:

    overview/index
    getting-started/index
    tutorials/index
    api
    development/index
    changelog

