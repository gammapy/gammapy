.. include:: ../references.txt

.. _makers:

***********************
makers - Data reduction
***********************

.. currentmodule:: gammapy.makers

Introduction
============

The `gammapy.makers` sub-package contains classes to perform data reduction tasks
from DL3 data to binned datasets. In the data reduction step the DL3 data is prepared for modeling and fitting,
by binning events into a counts map and interpolating the exposure, background,
psf and energy dispersion on the chosen analysis geometry.


Using `gammapy.makers`
======================

Gammapy tutorial notebooks that show examples using ``gammapy.makers``:

.. nbgallery::
 
   ../tutorials/api/makers.ipynb
   ../tutorials/starting/analysis_2.ipynb
   ../tutorials/analysis/3D/analysis_3d.ipynb
   ../tutorials/analysis/3D/simulate_3d.ipynb
   ../tutorials/analysis/1D/spectral_analysis.ipynb
   ../tutorials/analysis/1D/spectrum_simulation.ipynb

Other examples using background makers:

.. toctree::
    :maxdepth: 1

    fov
    reflected
    ring


Reference/API
=============

.. automodapi:: gammapy.makers
    :no-inheritance-diagram:
    :include-all-objects:
