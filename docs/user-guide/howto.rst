.. include:: ../references.txt

.. _how_to:

How To
======

This page contains short "how to" or "frequently asked question" entries for
Gammapy. Each entry is for a very specific task, with a short answer, and links
to examples and documentation.

If you're new to Gammapy, please check the :ref:`getting-started` section and
the :ref:`user_guide` and have a look at the list of :ref:`tutorials`.
The information below is in addition to those pages, it's not a complete list of
how to do everything in Gammapy.

Please give feedback and suggest additions to this page!


.. accordion-header::
    :id: collapseHowToOne
    :title: Spell and pronounce Gammapy

The recommended spelling is "Gammapy" as proper name. The recommended
pronunciation is [ɡæməpaɪ] where the syllable "py" is pronounced like
the english word "pie". You can listen to it `here <http://ipa-reader.xyz/?text=ˈ%C9%A1æməpaɪ&voice=Amy>`__.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToTwo
    :title: Select observations
    :link: ../tutorials/starting/analysis_2.html#defining-the-datastore-and-selecting-observations

The `~gammapy.data.DataStore` provides access to a summary table of all observations available.
It can be used to select observations with various criterion. You can for instance apply a cone search
or also select observations based on other information available using the `~gammapy.data.ObservationTable.select_observations` method.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToThree
    :title: Make an on-axis equivalent livetime map
    :link: ../tutorials/data/hess.html#on-axis-equivalent-livetime

The `~gammapy.data.DataStore` provides access to a summary table of all observations available.
It can be used to select observations with various criterion. You can for instance apply a cone search
or also select observations based on other information available using the `~gammapy.data.ObservationTable.select_observations` method.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToFour
    :title: Check IRFs
    :link: ../tutorials/data/cta.html#irfs

Gammapy offers a number of methods to explore the content of the various IRFs
contained in an observation. This is usually done thanks to their ``peek()``
methods.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToFive
    :title: Choose units for plotting

Units for plotting are handled with a combination of `matplotlib` and `astropy.units`.
The methods `ax.xaxis.set_units()` and `ax.yaxis.set_units()` allow
you to define the x and y axis units using `astropy.units`. Here is a minimal example:

.. code::

        import matplotlib.pyplot as plt
        from gammapy.estimators import FluxPoints
        from astropy import units as u

        filename = "$GAMMAPY_DATA/hawc_crab/HAWC19_flux_points.fits"
        fp = FluxPoints.read(filename)

        ax = plt.subplot()
        ax.xaxis.set_units(u.eV)
        ax.yaxis.set_units(u.Unit("erg cm-2 s-1"))
        fp.plot(ax=ax, sed_type="e2dnde")

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToSix
    :title: Compute source significance

Estimate the significance of a source, or more generally of an additional model
component (such as e.g. a spectral line on top of a power-law spectrum), is done
via a hypothesis test. You fit two models, with and without the extra source or
component, then use the test statistic values from both fits to compute the
significance or p-value. To obtain the test statistic, call
`~gammapy.modeling.Dataset.stat_sum` for the model corresponding to your two
hypotheses (or take this value from the print output when running the fit), and
take the difference. Note that in Gammapy, the fit statistic is defined as ``S =
- 2 * log(L)`` for likelihood ``L``, such that ``TS = S_0 - S_1``. See
:ref:`datasets` for an overview of fit statistics used.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToSeven
    :title: Compute cumulative significance
    :link: ../tutorials/analysis-1d/spectral_analysis.html#source-statistic

A classical plot in gamma-ray astronomy is the cumulative significance of a
source as a function of observing time. In Gammapy, you can produce it with 1D
(spectral) analysis. Once datasets are produced for a given ON region, you can
access the total statistics with the ``info_table(cumulative=True)`` method of
`~gammapy.datasets.Datasets`.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToEight
    :title: Implement a custom model
    :link: ../tutorials/api/models.html#implementing-a-custom-model

Gammapy allows the flexibility of using user-defined models for analysis.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToNine
    :title: Implement energy dependent spatial models
    :link: ../tutorials/api/models.html#models-with-energy-dependent-morphology

While Gammapy does not ship energy dependent spatial models, it is possible to define
such models within the modeling framework.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToTen
    :title: Model astrophysical source spectra

It is possible to combine Gammapy with astrophysical modeling codes, if they
provide a Python interface. Usually this requires some glue code to be written,
e.g. `~gammapy.modeling.models.NaimaSpectralModel` is an example of a Gammapy
wrapper class around the Naima spectral model and radiation classes, which then
allows modeling and fitting of Naima models within Gammapy (e.g. using CTA,
H.E.S.S. or Fermi-LAT data).

.. accordion-footer::

.. accordion-header::
    :id: collapseSHowToEleven
    :title: Model temporal profiles
    :link: ../tutorials/analysis-time/light_curve_simulation.html#fitting-temporal-models

Temporal models can be directly fit on available lightcurves,
or on the reduced datasets. This is done through a joint fitting of the datasets,
one for each time bin.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToOne
    :title: Improve fit convergence with constraints on the source position

It happens that a 3D fit does not converge with warning messages indicating that the
scanned positions of the model are outside the valid IRF map range. The type of warning message is:
::

    Position <SkyCoord (ICRS): (ra, dec) in deg
      (329.71693826, -33.18392464)> is outside valid IRF map range, using nearest IRF defined within

This issue might happen when the position of a model has no defined range. The minimizer
might scan positions outside the spatial range in which the IRFs are computed and then it gets lost.

The simple solution is to add a physically-motivated range on the model's position, e.g. within
the field of view or around an excess position. Most of the time, this tip solves the issue.
The documentation of the
`models sub-package <https://docs.gammapy.org/1.0/tutorials/api/models.html#modifying-model-parameters>`_
explains how to add a validity range of a model parameter.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToTwelve
    :title: Reduce memory budget for large datasets

When dealing with surveys and large sky regions, the amount of memory required might become
problematic, in particular because of the default settings of the IRF maps stored in the
`~gammapy.datasets.MapDataset` used for the data reduction. Several options can be used to reduce
the required memory:
- Reduce the spatial sampling of the `~gammapy.irf.PSFMap` and the `~gammapy.irf.EDispKernelMap`
using the `binsz_irf` argument of the `~gammapy.datasets.MapDataset.create` method. This will reduce
the accuracy of the IRF kernels used for model counts predictions.
- Change the default IRFMap axes, in particular the `rad_axis` argument of `~gammapy.datasets.MapDataset.create`
This axis is used to define the geometry of the `~gammapy.irf.PSFMap` and controls the distribution of error angles
used to sample the PSF. This will reduce the quality of the PSF description.
- If one or several IRFs are not required for the study at hand, it is possible not to build them
by removing it from the list of options passed to the `~gammapy.makers.MapDatasetMaker`.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToThirteen
    :title: Copy part of a data store

To share specific data from a database, it might be necessary to create a new data storage with
a limited set of observations and summary files following the scheme described in gadf_.
This is possible with the method `~gammapy.data.DataStore.copy_obs` provided by the
`~gammapy.data.DataStore`. It allows to copy individual observations files in a given directory
and build the associated observation and HDU tables.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToFourteen
    :title: Interpolate onto a different geometry
    :link: ../tutorials/api/maps.html#filling-maps-from-interpolation

To interpolate maps onto a different geometry use `~gammapy.maps.Map.interp_to_geom`.

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToFifteen
    :title: Suppress warnings

In general it is not recommended to suppress warnings from code because they
might point to potential issues or help debugging a non-working script. However
in some cases the cause of the warning is known and the warnings clutter the
logging output. In this case it can be useful to locally suppress a specific
warning like so:

.. testcode::

    from astropy.io.fits.verify import VerifyWarning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', VerifyWarning)
        # do stuff here

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToSixteen
    :title: Display a progress bar

Gammapy provides the possibility of displaying a
progress bar to monitor the advancement of time-consuming processes. To activate this
functionality, make sure that `tqdm` is installed and add the following code snippet
to your code:

.. testcode::

    from gammapy.utils import pbar
    pbar.SHOW_PROGRESS_BAR = True

The progress bar is available within the following:

* `~gammapy.analysis.Analysis.get_datasets` method

* `~gammapy.data.DataStore.get_observations` method

* The ``run()`` method from the ``estimator`` classes: `~gammapy.estimators.ASmoothMapEstimator`, `~gammapy.estimators.TSMapEstimator`, `~gammapy.estimators.LightCurveEstimator`

* `~gammapy.modeling.Fit.stat_profile` and `~gammapy.modeling.Fit.stat_surface` methods

* `~gammapy.scripts.download.progress_download` method

* `~gammapy.utils.parallel.run_multiprocessing` method

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToSeventeen
    :title: Change plotting style and color-blind friendly visualizations

As the Gammapy visualisations are using the library `matplotlib` that provides color styles, it is possible to change the
default colors map of the Gammapy plots. Using using the
`style sheet of matplotlib <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_, you
should add into your notebooks or scripts the following lines after the Gammapy imports:

.. code::

    import matplotlib.style as style
    style.use('XXXX')
    # with XXXX from `print(plt.style.available)`

Note that you can create your own style with matplotlib (see
`here <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ and
`here <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_)

The CTA observatory released a document describing best practices for **data visualisation in a way friendly to
color-blind people**:
`CTAO document <https://www.cta-observatory.org/wp-content/uploads/2020/10/CTA_ColourBlindness_BestPractices2.pdf>`_. To
use them, you should add into your notebooks or scripts the following lines after the Gammapy imports:

.. code::

    import matplotlib.style as style
    style.use('tableau-colorblind10')

or

.. code::

    import matplotlib.style as style
    style.use('seaborn-colorblind')

.. accordion-footer::

.. accordion-header::
    :id: collapseHowToEighteen
    :title: Add PHASE information to your data

For doing pulsar analysis, you must compute the phase associated
to each event and then create a new `EventList` and a new `Observation`.
Modifying the `EventList` of an `Observation` in-place is prohibited because of the
underlying lazy loading implemented in reading observations.
Code for computing phases is NOT provided within gammapy,
and you must use an external s/w like PINT or TEMPO2. For brevity,
this code example shows the only technical implementation
using a dummy phase column.

.. testcode::

    import numpy as np
    from gammapy.data import DataStore, Observation, EventList

    # read the observation
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs = datastore.obs(23523)

    # use the phase information - dummy in this example
    phase = np.random.random(len(obs.events.table))

    # create a new `EventList`
    table = obs.events.table
    table["PHASE"] = phase
    events_new = EventList(table)

    # copy the observation in memory, changing the events
    o2 = obs.copy(events=events_new, in_memory=True)

    # The new observation and the new events table can be serialised independently
    o2.write("new_obs.fits.gz")
    events_new.write("events.fits.gz", gti=obs.gti, overwrite=True)

.. accordion-footer::
