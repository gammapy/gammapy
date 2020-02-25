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
and parameter API. An overview of all the available models can be found in the :ref:`model-gallery`.


Getting Started
===============
In the following you will see how to fit spectral data in OGIP format. The
format is described at :ref:`gadf:ogip`. An example dataset is available in the
``$GAMMAPY_DATA`` repo. For a description of the available fit statstics see
:ref:`fit-statistics`.


The following example shows how to fit a power law simultaneously to two
simulated crab runs using the `~gammapy.modeling.Fit` class.

.. code-block:: python

    from gammapy.spectrum import SpectrumDatasetOnOff
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

- `Models Tutorial <../notebooks/models.html>`__
- `Modeling and Fitting <../notebooks/modeling.html>`__
- `analysis_3d.html <../notebooks/analysis_3d.html>`__
- `spectrum_analysis.html <../notebooks/spectrum_analysis.html>`__


Reference/API
=============

.. automodapi:: gammapy.modeling
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.modeling.models
    :no-inheritance-diagram:
    :include-all-objects:
