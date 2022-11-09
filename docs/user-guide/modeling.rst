.. _modeling:

Modeling and Fitting (DL4 to DL5)
=================================

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

Built-in models
---------------

Gammapy provides a large choice of spatial, spectral and temporal models.
You may check out the whole list of built-in models in the :ref:`model-gallery`.

Custom models
---------------

Gammapy provides an easy interface to :ref:`custom-model`.


Using gammapy.modeling
----------------------

.. minigallery:: gammapy.modeling.Fit
    :add-heading:

.. include:: ../references.txt

