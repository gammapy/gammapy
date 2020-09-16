.. include:: ../references.txt

.. _modeling:

*****************************
modeling - Models and fitting
*****************************

.. currentmodule:: gammapy.modeling

Introduction
============

`gammapy.modeling` contains all the functionality related to modeling and fitting
data. This includes spectral, spatial and temporal model classes, as well as the fit
and parameter API.

Assuming you have prepared your gamma-ray data as a set of
`~gammapy.datasets.Dataset` objects, and
stored one or more datasets in a `~gammapy.datasets.Datasets` container, you are
all set for modeling and fitting. Either via a YAML config file, or via Python
code, define the `~gammapy.modeling.models.Models` to use, which is a list of
`~gammapy.modeling.models.SkyModel` objects representing additive emission
components, usually sources or diffuse emission, although a single source can
also be modeled by multiple components if you want. The
`~gammapy.modeling.models.SkyModel` is a factorised model with a
`~gammapy.modeling.models.SpectralModel` component and a
`~gammapy.modeling.models.SpatialModel` component. Most commonly used models in
gamma-ray astronomy are built-in, see the :ref:`model-gallery`.
It is easy to create user-defined models and
datasets, Gammapy is very flexible.

The `~gammapy.modeling.Fit` class provides methods to fit, i.e. optimise
parameters and estimate parameter errors and correlations. It interfaces with a
`~gammapy.datasets.Datasets` object, which in turn is connected to a
`~gammapy.modeling.models.Models` object, which has a
`~gammapy.modeling.Parameters` object, which contains the model parameters.
Currently ``iminuit`` is used as modeling and fitting backend, in the future we
plan to support other optimiser and error estimation methods, e.g. from
``scipy``. Models can be unique for a given dataset, or contribute to multiple
datasets and thus provide links, allowing e.g. to do a joint fit to multiple
IACT datasets, or to a joint IACT and Fermi-LAT dataset. Many examples are given
in the tutorials.

Getting Started
===============
In the following you will see how to fit spectral data in OGIP format. The
format is described at :ref:`gadf:ogip`. An example dataset is available in the
``$GAMMAPY_DATA`` repo. For a description of the available fit statstics see
:ref:`fit-statistics`.


The following example shows how to fit a power law simultaneously to two
simulated crab runs using the `~gammapy.modeling.Fit` class.

.. code-block:: python

    from gammapy.datasets import SpectrumDatasetOnOff
    from gammapy.modeling import Fit
    from gammapy.modeling.models import PowerLawSpectralModel
    import matplotlib.pyplot as plt

    path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
    obs_1 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23523.fits")
    obs_2 = SpectrumDatasetOnOff.from_ogip_files(path + "pha_obs23592.fits")

    model = PowerLawSpectralModel(
        index=2,
        amplitude='1e-12  cm-2 s-1 TeV-1',
        reference='1 TeV',
    )

    obs_1.model = model
    obs_2.model = model

    fit = Fit([obs_1, obs_2])
    result = fit.run()

    model.parameters.covariance = result.parameters.covariance

You can check the fit results by looking at the result and model object:

.. code-block:: python

    >>> print(result)

        OptimizeResult

        backend    : minuit
        method     : minuit
        success    : True
        nfev       : 115
        total stat : 65.36
        message    : Optimization terminated successfully.


    >>> print(model)

        PowerLawSpectralModel

        Parameters:

               name     value     error        unit      min max frozen
            --------- --------- --------- -------------- --- --- ------
                index 2.781e+00 1.120e-01                nan nan  False
            amplitude 5.201e-11 4.965e-12 cm-2 s-1 TeV-1 nan nan  False
            reference 1.000e+00 0.000e+00            TeV nan nan   True

        Covariance:

               name     index   amplitude reference
            --------- --------- --------- ---------
                index 1.255e-02 3.578e-13 0.000e+00
            amplitude 3.578e-13 2.465e-23 0.000e+00
            reference 0.000e+00 0.000e+00 0.000e+00



Tutorials
=========


:ref:`tutorials` that show examples using ``gammapy.modeling``:

- `Models Tutorial <../tutorials/models.html>`__
- `Modeling and Fitting <../tutorials/modeling.html>`__
- `analysis_3d.html <../tutorials/analysis_3d.html>`__
- `spectrum_analysis.html <../tutorials/spectrum_analysis.html>`__


Reference/API
=============

.. automodapi:: gammapy.modeling
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.modeling.models
    :no-inheritance-diagram:
    :include-all-objects:
