.. include:: ../references.txt

.. _references:

Glossary and references
=======================

.. _glossary:



Glossary
--------

.. glossary::

    1D Analysis
      1D analysis or spectral analysis where data are reduced to a simple 1D
      geometry along the reconstructed energy axis. In Cherenkov astronomy,
      this is classically performed with an OFF background measurement, see WStat.

    3D Analysis
      3D analysis or cube analysis, where data are reduced to a 3D cube with
      spatial coordinates and energy axes [Stewart2009]_. In Gammapy, these cube are
      represented by `~gammapy.maps.Map` objects (see :ref:`maps`) and contained in a `~gammapy.datasets.MapDataset` object.

    Aeff
      Short for "effective area": it is the IRF representing the detector collection
      area. See :ref:`irf` and :ref:`irf-aeff`.

    Bkg
      Short for "background": it is the IRF representing the residual background rate
      per solid angle. See :ref:`irf` and :ref:`irf-bkg`.

    Cash
      The Cash statistic is a Poisson fit statistic usually used when signal and
      background can be modeled. It is defined as :math:`2 \times log(L)`. See
      :ref:`cash` in :ref:`fit statistics <fit-statistics>`.

    Dataset
      In Gammapy a dataset bundles the data, IRFs, model and a likelihood function.
      Based on the model and IRFs the predicted number of counts are computed and
      compared to the measured counts using the likelihood. See :ref:`datasets`

    DL3
      Short for "data level 3": it is used to mention or describe science-ready data, ie
      the event list (with basically a Time, an arrival direction and an energy) and the
      four IRFs (Aeff, EDisp, PSF, Bkg). See :ref:`data flow <data_flow>`.

    DL4
      Short for "data level 4": it is used to mention or describe binned-science data, ie
      N-dim maps (multi-dimensional histograms) of the spatial, temporal and/or spectral
      components of the DL3 data in instrumental units (e.g. counts). See :ref:`maps` and
      :ref:`data flow <data_flow>`.

    DL5
      Short for "data level 5": it is used to mention or describe advanced-science data, ie
      N-dim maps of the spatial, temporal and/or spectral astrophysical quantities in
      physical units (e.g. cm-2 s-1). See :ref:`data flow <data_flow>`.

    DL6
      Short for "data level 6": it is used to mention or describe catalogs of sources,
      with their spectral and spatial properties. See :ref:`data flow <data_flow>`.

    EDisp
      Short for "energy dispersion": it is the IRF that represents the probability
      of measuring a given reconstructed energy as a function of the true photon
      energy. See :ref:`irf` and :ref:`irf-edisp`.

    Estimator
      Gammapy utility classes to compute astrophysical quantities based on a model.
      They are used for the computation of DL5 products from the DL4 ones. See
      :ref:`modeling` and the :ref:`data flow <data_flow>`.

    FoV
      Short for "field of view": it indicates the angular aperture (sometimes also the
      solid angle) on the sky that is visible by the instrument with a single pointing.

    FoV Background
      Background estimation method typically used for the 3D analysis. See :ref:`makers`.

    HLI
      Short for "high level interface: this API aims to realize most of the analysis
      types using YAML configuration files. See :ref:`analysis`

    GADF
      Short for "Gamma Astro Data Format".
      The `open initiative GADF <https://gamma-astro-data-formats.readthedocs.io/en/v0.2/>`_
      provides a Data Format for gamma-ray data that is currently used by many :term:`IACT` experiments
      and by `HAWC`_. Gammapy I/O functions are compliant with this format.

    GTI
      Short for "good time interval": it indicates a continuous time interval of data
      acquisition. In the GADF DL3 format, it also represents a time interval in which the IRFs are
      supposed to be constant.

    IACT
      Short for "Imaging Atmospheric Cherenkov Technique", as in used for e.g. `CTAO`_.

    IRF
      Short for "instrument response function": they are used to model the probability
      to detect a photon with a number of measured characteristics. See :ref:`irf`.

    Joint Analysis
      A joint fit across multiple datasets implies that each dataset is handled
      independently during the data reduction stage,
      and the statistics combined during the likelihood fit.
      The likelihood is computed for each dataset and summed to get
      the total fit statistic. See :ref:`joint`

    Maker
      Gammapy utility classes performing data reduction of the DL3 data to binned datasets (DL4).
      See :ref:`makers` and the :ref:`data flow <data_flow>`.

    MCMC
      Short for "Markov chain Monte Carlo". See the
      `following recipe <https://gammapy.github.io/gammapy-recipes/_build/html/notebooks/mcmc-sampling-emcee/mcmc_sampling.html>`_.

    MET
      Short for "mission elapsed time". See also :ref:`MET_definition` in :ref:`time_handling`.

    Migration
        This is a quantity described by the true and reconstructed energy: :math:`\mu = E_{\rm{reco}} / E_{\rm{true}}`

    Offset
        The angular separation from the field of view (FoV) center.

    PSF
      Short for "point spread function": it is the IRF representing the probability density of the angular separation
      between true and reconstructed directions. See :ref:`irf` and :ref:`irf-psf`.

    Reco Energy
      The reconstructed (or measured) energy (often written ``e_reco``) is the energy of
      the measured photon by contrast with its actual true energy. Measured quantities
      such as counts are represented along a reco energy axis.

    Reflected Background
      Background estimation method typically used for spectral analysis. See :ref:`makers`.

    Ring Background
      Background estimation method typically used for image analysis. See :ref:`makers`.

    RoI
      Short for "region of interest": it indicates the spatial region in which the
      data are analyzed. In practice, at each energy it corresponds with the sky region
      in which the dataset mask is True.

    SED
      Short for "spectral energy distribution". For a spectral model or flux points
      object, the type of plot (e.g. :math:`dN/dE`, :math:`E^2\ dN/dE`) is typically adjusted
      through the ``sed_type`` quantity. See :ref:`sedtypes` for a list of options.


    Stacked Analysis
      In a stacked analysis individual observations are reduced to datasets which
      are then stacked to produce a single reduced dataset. The latter is then used
      to obtain physical information through model fitting. Some approximations must
      be made to perform dataset stacking (e.g. loss of individual background normalization,
      averaging of instrument responses, loss of information outside region of interest etc),
      but this can reduce very significantly the computing and memory cost. For details, see :ref:`stack`

    True Energy
      The true energy (often written ``e_true``) is the energy of the incident photon
      by contrast with the energy reconstructed by the instrument. Instrument response
      functions are represented along a true energy axis.

    Transit
      Path of a source over the sky in the course of one full sidereal day
      as seen by drift instruments like `HAWC`_ and `SWGO`_.

    TS
      Short for "test statistics". See :ref:`ts` and :ref:`fit-statistics`.

    WCD
      Short for "Water Cherenkov Detector", like the experiments `HAWC`_ or `SWGO`_.

    WStat
      The WStat is a Poisson fit statistic usually used for ON-OFF analysis. It is
      based on the profile likelihood method where the unknown background parameters
      are marginalized. See :ref:`wstat` in :ref:`fit statistics <fit-statistics>`.

.. _publications:

References
----------

This is the bibliography containing the literature references for the implemented methods
referenced from the Gammapy docs.

.. [Abdalla2018] `H.E.S.S. Collaboration (2018) <https://www.aanda.org/articles/aa/abs/2018/04/aa32098-17/aa32098-17.html>`_,
    "The H.E.S.S. Galactic plane survey"

.. [Albert2007] `Albert et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007NIMPA.583..494A>`_,
   "Unfolding of differential energy spectra in the MAGIC experiment",

.. [Berge2007] `Berge et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007A%26A...466.1219B>`_,
   "Background modelling in very-high-energy gamma-ray astronomy"

.. [Cash1979] `Cash (1979) <https://ui.adsabs.harvard.edu/abs/1979ApJ...228..939C>`_,
   "Parameter estimation in astronomy through application of the likelihood ratio"

.. [Cousins2007] `Cousins et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007physics...2156C>`_,
   "Evaluation of three methods for calculating statistical significance when incorporating a
   systematic uncertainty into a test of the background-only hypothesis for a Poisson process"

.. [Feldman1998] `Feldman & Cousins (1998) <https://ui.adsabs.harvard.edu/abs/1998PhRvD..57.3873F>`_,
   "Unified approach to the classical statistical analysis of small signals"

.. [Lafferty1994] `Lafferty & Wyatt (1994) <https://ui.adsabs.harvard.edu/abs/1995NIMPA.355..541L>`_,
   "Where to stick your data points: The treatment of measurements within wide bins"

.. [LiMa1983] `Li & Ma (1983) <https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L>`_,
   "Analysis methods for results in gamma-ray astronomy"

.. [Meyer2010] `Meyer et al. (2010) <https://ui.adsabs.harvard.edu/abs/2010A%26A...523A...2M>`_,
   "The Crab Nebula as a standard candle in very high-energy astrophysics"

.. [Mohrmann2019] `Mohrmann et al. (2019) <https://www.aanda.org/articles/aa/abs/2019/12/aa36452-19/aa36452-19.html>`_,
    "Validation of open-source science tools and background model construction in γ-ray astronomy"

.. [Naurois2012] `de Naurois (2012) <http://inspirehep.net/record/1122589>`_,
   "Very High Energy astronomy from H.E.S.S. to CTA. Opening of a new astronomical window on the non-thermal Universe",

.. [Piron2001] `Piron et al. (2001) <https://ui.adsabs.harvard.edu/abs/2001A%26A...374..895P>`_,
   "Temporal and spectral gamma-ray properties of Mkn 421 above 250 GeV from CAT observations between 1996 and 2000",

.. [Rolke2005] `Rolke et al. (2005) <https://ui.adsabs.harvard.edu/abs/2005NIMPA.551..493R>`_,
   "Limits and confidence intervals in the presence of nuisance parameters",

.. [Stewart2009] `Stewart (2009) <https://ui.adsabs.harvard.edu/abs/2009A%26A...495..989S>`_,
   "Maximum-likelihood detection of sources among Poissonian noise"
