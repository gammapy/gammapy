.. include:: references.txt

.. image:: _static/gammapy_banner.png
    :width: 400px

.. Should we add a known issues section at the end?

Gammapy
-------

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


.. _gammapy_intro:
.. toctree::
    :caption: User Guide
    :maxdepth: 1

    overview/index
    getting-started/index
    tutorials/index
    modeling/gallery/index
    howto
    references
    changelog
    List of releases <https://gammapy.org/news.html#releases>


.. _gammapy_package:

.. toctree::
    :caption: Gammapy Package
    :maxdepth: 1

    analysis/index
    data/index
    makers/index
    datasets/index
    modeling/index
    estimators/index

    irf/index
    maps/index
    catalog/index

    astro/index
    scripts/index
    stats/index
    visualization/index
    utils/index


.. _gammapy_dev:

.. toctree::
    :caption: Developer Documentation
    :titlesonly:
    :maxdepth: 1

    development/index
