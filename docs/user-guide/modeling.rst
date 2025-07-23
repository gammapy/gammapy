.. _modeling:

Modeling and Fitting (DL4 to DL5)
=================================

Modeling
--------

`gammapy.modeling` contains all the functionalities related to modeling and fitting
data. This includes spectral, spatial and temporal model classes, as well as the fit
and parameter API.

Assuming you have prepared your gamma-ray data as a set of `~gammapy.datasets.Dataset` objects, and
stored one or more datasets in a `~gammapy.datasets.Datasets` container, you are all set for modeling
and fitting. Either via a YAML config file, or via Python code, define the
`~gammapy.modeling.models.Models` to use, which is a list of `~gammapy.modeling.models.SkyModel` objects
representing additive emission components, usually sources or diffuse emission, although a single source
can also be modeled by multiple components if you want. The `~gammapy.modeling.models.SkyModel` is a
factorised model with a `~gammapy.modeling.models.SpectralModel` component, a
`~gammapy.modeling.models.SpatialModel` component and a `~gammapy.modeling.models.TemporalModel`,
depending of the type of `~gammapy.datasets.Datasets`.

Most commonly used models in gamma-ray astronomy are built-in, see the :ref:`model-gallery`.
It is easy to sum models, create compound spectral models (see
:doc:`/user-guide/model-gallery/spectral/plot_compound`), or to create user-defined models (see
:ref:`custom-model`). Gammapy is very flexible!

Models can be unique for a given dataset, or contribute to multiple datasets and thus provide links,
allowing e.g. to do a joint fit to multiple :term:`IACT` datasets, or to a joint :term:`IACT` and
`Fermi-LAT`_ dataset (see :doc:`/tutorials/analysis-3d/analysis_mwl`).


Built-in models
^^^^^^^^^^^^^^^

Gammapy provides a large choice of spatial, spectral and temporal models.
You may check out the whole list of built-in models in the :ref:`model-gallery`.

Custom models
^^^^^^^^^^^^^

Gammapy provides an easy interface to :ref:`custom-model`.


Using gammapy.modeling
^^^^^^^^^^^^^^^^^^^^^^

.. minigallery::

    ../examples/tutorials/api/models.py
    ../examples/tutorials/api/model_management.py
    ../examples/tutorials/api/priors.py


Fitting
-------

Gammapy offers several statistical methods to estimate 'best' parameters from the data:

- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori estimation (MAP)
- Bayesian Inference


Maximum Likelihood Estimation (MLE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method permits to estimate parameters without having some knowledge on their probability distribution.
Their "prior" distributions are then uniform in the region of interest. The estimation is achieved by maximizing
a likelihood function by finding the models' parameters for which the observed data have the highest joint probability.
For this method, the "probability" is equivalent to "frequency". It is a **Frequentist inference** of parameters from
sample-data, permitting hypothesis testing, confidence intervals and confidence limits. Commonly, this method is
a "fit" on the data.

The `~gammapy.modeling.Fit` class provides methods to fit, i.e. optimise parameters and estimate parameter
errors and correlations. It interfaces with a `~gammapy.datasets.Datasets` object, which in turn is
connected to a `~gammapy.modeling.models.Models` object, which has a `~gammapy.modeling.Parameters`
object, which contains the model parameters.

Three different fitting backend are offered:

- `minuit <https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit>`__ (used by default),
- `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__,
- `sherpa <http://cxc.cfa.harvard.edu/sherpa/methods/index.html>`__.

``Sherpa`` is not installed by default, but this is quite easy (see :ref:`quickstart-setup`).

The tutorial :doc:`/tutorials/api/fitting` describes in detail the API.

Maximum A Posteriori estimation (MAP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For some physics use cases, some parameters of the models might have some astrophysical constraints, e.g.
the usual case of positive flux, spectral index range. These knowledge can be used when estimating
parameters. To do so, we incorporate a `~gammapy.modeling.models.Prior` density over the quantities
one wants to estimate and the `~gammapy.modeling.Fit` class is used to determine the best parameters by
by regularizing the maximum a posteriori likelihood (a combination of the data likelihood term and of the prior
term).

With the MAP estimation, one can also realise hypothesis testing, compute confidence intervals and confidence
limits.

The tutorial :doc:`/tutorials/api/priors` describes in detail this estimation method.

Bayesian Inference
^^^^^^^^^^^^^^^^^^

This Bayesian method uses prior knowledge on each models' parameters, as `~gammapy.modeling.models.Prior`,
in order to estimate the posterior probabilities for each set of parameters values. They are
traditionally visualised in the form of corner plots of the parameters. This inference of the best
a posteriori parameters are not associated to the "best model" in the Frequentist sense, but rather to
the most probable given a set of parameters' priors.

The Bayesian Inference is using dedicated tools to estimate posterior likelihood probabilities, e.g.
the Markov Chain Monte Carlo (MCMC) approach or the Nested sampling (NS) approach that we recommend for
Gammapy.

This method is quite powerful in case of non-Gaussian degeneracies, larger number of parameters, or to
map likelihood landscapes with multiple solutions (local maxima in which a classical fit would fall into).
However, the computation time to make Bayesian Inference is generally larger than for the Maximum
Likelihood Estimation using ``iminuit``.

The tutorial :doc:`/tutorials/api/nested_sampling_Crab` describes in detail this estimation
method.


Using gammapy.modeling
^^^^^^^^^^^^^^^^^^^^^^

.. minigallery::

    ../examples/tutorials/api/fitting.py
    ../examples/tutorials/analysis-1d/spectral_analysis.py
    ../examples/tutorials/analysis-3d/analysis_3d.py
    ../examples/tutorails/analysis-3d/analysis_mwl.py
    ../examples/tutorials/analysis-1d/sed_fitting.py
    ../examples/tutorials/api/priors.py
    ../examples/tutorials/api/nested_sampling_Crab.py

.. include:: ../references.txt

