.. include:: ../../references.txt

.. _makers:

Data reduction (DL3 to DL4)
===========================

The `gammapy.makers` sub-package contains classes to perform data reduction tasks
from DL3 data to binned datasets. In the data reduction step the DL3 data is prepared for modeling and fitting,
by binning events into a counts map and interpolating the exposure, background,
psf and energy dispersion on the chosen analysis geometry.

Background estimation
---------------------

.. toctree::
    :maxdepth: 1

    fov
    reflected
    ring


Safe data range definition
--------------------------

The definition of a safe data range is done using the `~gammapy.makers.SafeMaskMaker` or manually.


Using gammapy.makers
--------------------

.. minigallery::

    ../examples/tutorials/details/makers.py


.. minigallery::
    :add-heading: Examples using `~gammapy.makers.SpectrumDatasetMaker`

    ../examples/tutorials/analysis-1d/spectral_analysis.py
    ../examples/tutorials/analysis-1d/spectral_analysis_rad_max.py
    ../examples/tutorials/analysis-1d/extended_source_spectral_analysis.py


.. minigallery::
    :add-heading: Examples using `~gammapy.makers.MapDatasetMaker`

    ../examples/tutorials/starting/analysis_1.py
    ../examples/tutorials/data/hawc.py

