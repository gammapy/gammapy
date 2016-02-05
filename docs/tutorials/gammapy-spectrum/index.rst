.. include:: ../references.txt

.. _tutorials-gammapy-spectrum:

Spectral fitting with ``gammapy-spectrum``
==========================================

In this tutorial you will perform a spectra fit starting from an event list
and a set of 2D IRFs. The data used in this tutorial consist of 2 simulated
Crab observations with the H.E.S.S. array. They are available in the gammapy-extra
repository.

.. _tutorials-gammapy-spectrum-extract:

Extracting 1D spectral information
----------------------------------

The first step on our way to the Crab spectrum is generating 1D spectral
information, i.e. an on counts vector, and off counts vector, an effective area
vector, and an energy dispersion matrix. This is done with
``gammapy-spectrum extract``. Below you find an example config file:


.. literalinclude:: ./spectrum_extraction_example.yaml
    :language: yaml
    :linenos:


In Detail:

* Line 4-7 : Specify the DataStore to take the data from,
  a list of observations (list or filename),
  and the number of observations to analyse (0: all observations)
* Line 9-13 : Define reconstructed energy binning, the true energy binning is
  derived from the IRF files.
* Line 15-20 : Define region to extract the spectrum from (on region)
* Line 22-25 : Choose background estimation method.
  For available methods see :ref:`spectrum_background_method`
* Line 27-28 : Specify fits map containing exclusion regions. Here a map
  excluding all TevCat sources is used.
* Line 30-33 : Define ouput folder and files.

Note that at the moment it is necessary to create OGIP files at the spectrum
extraction step, because they will be used as input for the spectral fit (and
cannot be transferred in-memory).

To run the spectral extraction copy the above config file to your machine,
to e.g. ``crab_config.yaml`` and run

.. code-block:: bash

   gammapy-spectrum extract crab_config.yaml


This creates the folder ``crab_analysis`` in you current working directory. In
this folder you find all the generated OGIP file (in the folder ``ogip_data``),
as well the ``spectrum_stats.yaml`` file holding some parameters (TODO: link to
description). You can examine this file as explained in
:ref:`tutorials-gammapy-spectrum-examine`. Furthermore, there is an
``observations.fits`` file, which holds `~gammapy.data.ObservationTable` can
serves as input for ``gammapy-spectrum fit``

``gammapy-spectrum extract --interactive`` will drop you into an IPython session
  after the extraction step, where you can interactively look at all analysis
  results. This is illustrated in this IPython notebook (TODO: example).

Fitting a spectral model
------------------------

After having generated OGIP file in the section above, you could in principle
use fitting tools like XSPEC or Sherpa. There is, however,
the ``gammapy-spectrum fit`` command line tool which provides some convenience
for most use cases. This is an example config file

.. literalinclude:: ./spectrum_fit_example.yaml
    :language: yaml
    :linenos:

In Detail:

* Line 4 : Directory to write results to
* Line 5 : Input observation table, here we use the one created during
  :ref:`tutorials-gammapy-spectrum-extract`.
* Line 6 : Fitting backend
* Line 7-9 : Spectral model specifications
* Line 10 : Output file

Append this config file to your ``crab_config.yaml`` file and run the fit via

.. code-block:: bash

   gammapy-spectrum extract crab_config.yaml


``gammapy-spectrum fit`` also has an ``--interactive`` flag as shown in this
IPython notebook (TODO: example).

This creates the file ``fitresult.yaml`` in the ``crab_analysis`` folder.


.. _tutorials-gammapy-spectrum-examine:

Examining all results
---------------------

``gammapy-spectrum`` features two tools to examine and compare fit results.

* ``gammapy-spectrum display`` takes any number of results files as arguments
  and print a comparison table. The ``--browser`` flag shows this table in your
  browser.


.. code-block:: bash

    gammapy-spectrum display crab_analysis/*.yaml
                 analysis             index index_err       norm          norm_err    reference reference_err e_min e_max n_on n_off alpha excess
                                                      1 / (cm2 keV s) 1 / (cm2 keV s)    keV         keV       TeV   TeV
    --------------------------------- ----- --------- --------------- --------------- --------- ------------- ----- ----- ---- ----- ----- ------
        crab_analysis/fitresults.yaml  2.31     0.126        2.91e-20        3.62e-21     1e+09             0  1.03  10.1   --    --    --     --
    crab_analysis/spectrum_stats.yaml    --        --              --              --        --            --    --    --  416   414 0.149    354


* ``gammapy-spectrum plot`` can be used to plot fit results.


.. code-block:: bash

    gammapy-spectrum plot crab_analysis/fitresults.yaml --flux_unit 'cm-2 s-1 TeV-1'


.. image:: crab_plot.png




