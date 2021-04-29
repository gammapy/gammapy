.. include:: ../references.txt

.. _datasets:

***************************
datasets - Reduced datasets
***************************

.. currentmodule:: gammapy.datasets

Introduction
============

The `gammapy.datasets` sub-package contains classes to handle reduced
gamma-ray data for modeling and fitting.

The `Dataset` class bundles reduced data, IRFs and model to perform
likelihood fitting and joint-likelihood fitting.
All datasets contain a `~gammapy.modeling.models.Models` container with one or more
`~gammapy.modeling.models.SkyModel` objects that represent additive emission
components.

To model and fit data in Gammapy, you have to create a
`~gammapy.datasets.Datasets` container object with one or multiple
`~gammapy.datasets.Dataset` objects. Gammapy has built-in support to create and
analyse the following datasets: `~gammapy.datasets.MapDataset`,
`~gammapy.datasets.MapDatasetOnOff`, `~gammapy.datasets.SpectrumDataset`,
`~gammapy.datasets.SpectrumDatasetOnOff` and
`~gammapy.datasets.FluxPointsDataset`.

The map datasets represent 3D cubes (`~gammapy.maps.WcsNDMap` objects) with two
spatial and one energy axis. For 2D images the same map objects and map datasets
are used, an energy axis is present but only has one energy bin. The
`~gammapy.datasets.MapDataset` contains a counzts map, background is modeled with a
`~gammapy.modeling.models.BackgroundModel`, and the fit statistic used is
``cash``. The `~gammapy.datasets.MapDatasetOnOff` contains on and off count maps,
background is implicitly modeled via the off counts map, and the ``wstat`` fit
statistic.

The spectrum datasets represent 1D spectra (`~gammapy.maps.RegionNDMap`
objects) with an energy axis. There are no spatial axes or information, the 1D
spectra are obtained for a given on region. The
`~gammapy.datasets.SpectrumDataset` contains a counts spectrum, background is
modeled with a `~gammapy.maps.RegionNDMap`, and the fit statistic used is
``cash``. The `~gammapy.datasets.SpectrumDatasetOnOff` contains on on and off
count spectra, background is implicitly modeled via the off counts spectrum, and
the ``wstat`` fit statistic. The `~gammapy.datasets.FluxPointsDataset` contains
`~gammapy.estimatorsFluxPoints` and a spectral model, the fit statistic used is
``chi2``.

Note that in Gammapy, 2D image analyses are done with 3D cubes with a single
energy bin, e.g. for modeling and fitting,
see the `2D map analysis tutorial <./tutorials/image_analysis.html>`__.


To analyse multiple runs, you can either stack the datasets together, or perform
a joint fit across multiple datasets.

.. _stack:

Stacking Multiple Datasets
==========================

Stacking datasets implies that the counts, background and reduced IRFs from all the
runs are binned together to get one final dataset for which a likelihood is
computed during the fit. Stacking is often useful to reduce the computation effort while
analysing multiple runs.


The following table  lists how the individual quantities are handled during stacking.
Here, :math:`k` denotes a bin in reconstructed energy,
:math:`l` a bin in true energy and
:math:`j` is the dataset number

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Dataset attribute
     - Behaviour
     - Implementation
   * - ``livetime``
     - Sum of individual livetimes
     - :math:`\overline{t} = \sum_j t_j`
   * - ``mask_safe``
     - True if the pixel is included in the safe data range.
     - :math:`\overline{\epsilon_k} = \sum_{j} \epsilon_{jk}`
   * - ``mask_fit``
     - Dropped
     -
   * - ``counts``
     - Summed in the data range defined by `mask_safe`
     - :math:`\overline{\mathrm{counts}_k} = \sum_j \mathrm{counts}_{jk} \cdot \epsilon_{jk}`
   * - ``background``
     - Summed in the data range defined by `mask_safe`
     - :math:`\overline{\mathrm{bkg}_k} = \sum_j \mathrm{bkg}_{jk} \cdot \epsilon_{jk}`
   * - ``exposure``
     - Summed in the data range defined by `mask_safe`
     -  :math:`\overline{\mathrm{exposure}_l} = \sum_{j} \mathrm{exposure}_{jl} \cdot \sum_k \epsilon_{jk}`
   * - ``psf``
     - Exposure weighted average
     - :math:`\overline{\mathrm{psf}_l} = \frac{\sum_{j} \mathrm{psf}_{jl} \cdot \mathrm{exposure}_{jl}} {\sum_{j} \mathrm{exposure}_{jl}}`
   * - ``edisp``
     - Exposure weighted average, with mask on reconstructed energy
     - :math:`\overline{\mathrm{edisp}_{kl}} = \frac{\sum_{j}\mathrm{edisp}_{jkl} \cdot \epsilon_{jk} \cdot \mathrm{exposure}_{jl}} {\sum_{j} \mathrm{exposure}_{jl}}`
   * - ``gti``
     - Union of individual `gti`
     -

For the model evaluation, an important factor that needs to be accounted for is
that the energy threshold changes between obseravtions.
With the above implementation using a `~gammapy.irf.EDispersionMap`,
the `npred` is conserved,
ie, the predicted number of counts on the stacked
dataset is the sum expected by stacking the `npred` of the individual runs,

The following plot shows the individual and stacked energy dispersion kernel and `npred`  for two `SpectrumDataset`

.. plot:: datasets/plot_stack.py

.. note::
    - A stacked analysis is reasonable only when adding runs taken by the same instrument.
    - Stacking happens in-place, ie, ``dataset1.stack(dataset2)`` will overwrite ``dataset1``
    - To properly handle masks, it is necessary to stack onto an empty dataset.
    - Stacking only works for maps with equivalent geometry.
      Two geometries are called equivalent if one is exactly the same as, or can be obtained
      from a cutout of, the other.



.. _joint:
Joint Analysis
==============

An alternative to stacking datasets is a joint fit across all the datasets.
For a definition, see :ref:`glossary`.

The totat fit statistic of datasets is the sum of the
fit statistic of each dataset. Note that this is **not** equal to the
stacked fit statistic.

A joint fit usually allows a better modeling of the background because
the background model parameters can be fit for each dataset simultaneously
with the source models. However, a joint fit is, performance wise,
very computationally intensive.
The fit convergence time increases non-linearly with the number of datasets to be fit.
Moreover, depending upon the number of parameters in the background model,
even fit convergence might be an issue for a large number of datasets.

To strike a balance, what might be a practical solution for analysis of many runs is to
stack runs taken under similar conditions and then do a joint fit on the stacked datasets.



Reference/API
=============

.. automodapi:: gammapy.datasets
    :no-inheritance-diagram:
    :include-all-objects:
