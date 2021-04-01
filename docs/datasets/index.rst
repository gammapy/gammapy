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

Stacking Multiple Datasets
==========================

Stacking datasets implies that the counts, background and reduced IRFs from all the
runs are binned together to get one final dataset for which a likelihood is
computed during the fit.

Counts, background and exposure (lying outside the masked regions are simply summed),
while for energy dispersion and point spread function an avergae is computed after
weighing by the exposure.

For the model evaluation, an important factor that needs to be accounted for is
that the energy threshold changes between obseravtions.
To ensure that the npred (ie, the predicted number of counts) on the stacked
dataset is the sum expected by stacking the npred of the individual runs,
a `~gammapy.irf.EDispersionMap` is used, which contains the
The mask_safe from each dataset is applied on the respective reconstructed energy axis
of the energy dispersion matrix, and the masked matrices are combined.
Values lying outside the safe mask of each dataset are lost.

Stacking of multiple datasets is implemented as follows.
Here, :math:`k` denotes a bin in reconstructed energy,
:math:`l` a bin in true energy and
:math:`j` is the dataset number


================= ================================== ==================================================================================
Dataset attribute Behaviour                          Implementation
================= ================================== ==================================================================================

``livetime``.        Sum of individual livetimes         :math:`\overline{t} = \sum_j t_j`

``mask_safe``        Pixels added with `OR` operation     :math:`\overline{\epsilon_k} = \sum_{j} \epsilon_{jk}`

``mask_fit``         Dropped

``counts``           Summed outside exclusion region.     :math:`\overline{\mathrm{counts}_k} = \sum_j \mathrm{counts}_{jk} \cdot \epsilon_{jk}`

``background``       Summed outside exclusion region.    :math:`\overline{\mathrm{bkg}_k} = \sum_j \mathrm{bkg}_{jk} \cdot \epsilon_{jk}`

``exposure``         Summed outside spatial exclusion region    :math:`\overline{\mathrm{exposure}_l} = \sum_{j}{\mathrm{exposure}_{jl} \cdot \sum_k \epsilon_{jk}}`

``psf``              Exposure weighted average                 :math:`\overline{\mathrm{psf}_l} = \frac{ \sum_{j}{\mathrm{psf}_{jl} \cdot \mathrm{exposure}_l} {\sum_{j} \cdot \mathrm{exposure}_l}`

``edisp``            Exposure weighted average, with mask on reco energy :math:`\overline{\mathrm{edisp}_kl} = \frac{ \sum_{j}{\mathrm{edisp}_{jkl} \cdot \epsilon_{jk} \cdot \mathrm{exposure}_l} {\sum_{j} \cdot \mathrm{exposure}_l}`

``gti``              Union of individual `gti`

================= ================================== ==================================================================================



It is important to keep in mind that:

- Stacking happens in-place, ie, dataset1.stack(dataset2) will overwrite dataset
- To properly handle masks, it is necessary to stack onto an empty dataset.
- Stacking only works for maps with equivalent geometry.
 Two geometries are called equivalent if one is exactly the same as,
 or can be obtained from a cutout of, the other.
- A stacked analysis is reasonable only when adding runs taken by the same instrument.


Joint Analysis
==============

A joint fit across multiple datasets implies that each dataset
is handled independently during the data reduction stage,
and the statistics combined during the likelihood fit.
The likelihood is computed for each dataset and summed to get
the total fit statistic.

The totat fit statistic of datasets is the sum of the
fit statistic of each dataset. Note that this is **not** equal to the
stacked fit statistic.


Reference/API
=============

.. automodapi:: gammapy.datasets
    :no-inheritance-diagram:
    :include-all-objects:
