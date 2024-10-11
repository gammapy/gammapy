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
astronomy built on Numpy, Scipy and Astropy. **It is the core library for the** `CTAO`_ **Science Tools**
but can also be used to analyse data from existing imaging atmospheric Cherenkov telescopes
(IACTs), such as `H.E.S.S.`_, `MAGIC`_ and `VERITAS`_. It also provides some support
for `Fermi-LAT`_ and `HAWC`_ data analysis.

.. grid:: 1 2 2 2
    :gutter: 2
    :class-container: sd-text-center

    .. grid-item-card:: Getting started
        :img-top: _static/index_getting_started.svg
        :class-card: intro-card 
        :columns: 12 6 6 6
        :shadow: md

        New to *Gammapy*? Check out the getting started documents. They contain information
        on how to install and start using *Gammapy* on your local desktop computer.

        +++

        .. button-ref:: getting-started
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the quickstart docs
 
    .. grid-item-card:: User guide
        :img-top: _static/index_user_guide.svg
        :class-card: intro-card 
        :columns: 12 6 6 6
        :shadow: md

        The user guide provide in-depth information on the
        key concepts of Gammapy with useful background information and explanation,
        as well as tutorials in the form of Jupyter notebooks.

        +++

        .. button-ref:: user_guide
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the user guide
 
    .. grid-item-card:: API reference
        :img-top: _static/index_api.svg
        :class-card: intro-card 
        :columns: 12 6 6 6
        :shadow: md
 
        The reference guide contains a detailed description of
        the Gammapy API. The reference describes how the methods work and which parameters can
        be used. It assumes that you have an understanding of the key concepts.
 
        +++
 
        .. button-ref:: api-ref
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:
 
            To the reference guide
 
    .. grid-item-card:: Developer guide
        :img-top: _static/index_contribute.svg
        :class-card: intro-card 
        :columns: 12 6 6 6
        :shadow: md
 
        Saw a typo in the documentation? Want to improve
        existing functionalities? The contributing guidelines will guide
        you through the process of improving Gammapy.
 
        +++
 
        .. button-ref:: dev_intro
            :ref-type: ref
            :click-parent:
            :color: secondary
            :expand:

            To the developer guide
 

.. toctree::
    :titlesonly:
    :hidden:

    getting-started/index
    user-guide/index
    tutorials/index
    api-reference/index
    development/index
    release-notes/index
