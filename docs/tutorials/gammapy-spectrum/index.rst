.. include:: ../../references.txt

.. _tutorials-gammapy-spectrum:

Spectral fitting with ``gammapy-spectrum``
==========================================

In this tutorial you will perform a spectra fit using 4 simulated
Crab observations with the H.E.S.S. array. They data is available in the
gammapy-extra repo.

Running the fit
---------------

Create a new folder on your machine, e.g. `gammapy_crab_analysis` and put the
following :download:`example config file <./spectrum_analysis_example.yaml>`
into this folder

.. code-block:: bash

    mkdir gammapy_crab_analysis
    cd gammapy_crab_analysis
    wget http://docs.gammapy.org/en/latest/_downloads/spectrum_analysis_example.yaml

Now run

.. code-block:: bash

    $ gammapy-spectrum all spectrum_analysis_example.yaml

and check out the result of the spectral fit

.. code-block:: bash

    $ gammapy-spectrum display

                    Spectrum Stats
                    --------------
    n_on n_off alpha n_bkg excess energy_range [2]
                                        TeV
    ---- ----- ----- ----- ------ ----------------
     785   736 0.155   114    671    0.567 .. 99.1

                    Spectral Fit
                    ------------
     model   index index_err      norm         norm_err    reference reference_err fit_range [2]   flux[1TeV]   flux_err[1TeV]
                             1 / (m2 s TeV) 1 / (m2 s TeV)    TeV         TeV           TeV      1 / (m2 s TeV) 1 / (m2 s TeV)
    -------- ----- --------- -------------- -------------- --------- ------------- ------------- -------------- --------------
    PowerLaw  2.27    0.0486       2.34e-07       1.21e-08         1             0  0.722 .. 102       2.34e-07              0

To plot the fitted spectrum use

.. code-block:: bash

    $ gammapy-spectrum plot
    TODO: Implement

Results files
-------------
To get a better idea of what happend in the analysis have a look
at your analysis folder

.. code-block:: bash

    $ tree .
    .
    ├── fit_result_PowerLaw.yaml
    ├── observation_table.fits
    ├── ogip_data
    │   ├── arf_run23523.fits
    │   ├── arf_run23526.fits
    │   ├── arf_run23559.fits
    │   ├── arf_run23592.fits
    │   ├── bkg_run23523.fits
    │   ├── bkg_run23526.fits
    │   ├── bkg_run23559.fits
    │   ├── bkg_run23592.fits
    │   ├── pha_run23523.pha
    │   ├── pha_run23526.pha
    │   ├── pha_run23559.pha
    │   ├── pha_run23592.pha
    │   ├── rmf_run23523.fits
    │   ├── rmf_run23526.fits
    │   ├── rmf_run23559.fits
    │   └── rmf_run23592.fits
    ├── spectrum_analysis_example.yaml
    └── total_spectrum_stats.yaml


``gammapy-spectrum`` has created a folder ``ogip_data`` holding the extracted
spectrum for each observation and an ``observation_table.fits`` with meta
information about each observation. The statistics of the total, i.e. stacked
spectrum is stored in ``total_spectrum_stats.yaml``

.. code-block:: bash

    $ cat total_spectrum_stats.yaml
    spectrum:
      alpha: 0.15508291527313264
      energy_range:
        max: 99.0831944892862
        min: 0.567081905662225
        unit: TeV
      excess: 670.8589743589744
      n_off: 736
      n_on: 785

The result of the spectral fit is contained in ``fit_result_PowerLaw.yaml``

.. code-block:: bash

    $ cat fit_result_PowerLaw.yaml
    fit_result:
      fit_range:
         max: 101.7654126750981
        min: 0.7220744794075009
        unit: TeV
      fluxes:
        1TeV:
          error: 0.0
          unit: 1 / (m2 s TeV)
          value: 2.3400361534892314e-07
      parameters:
        index:
          error: 0.04861329659992716
          unit: ''
          value: 2.269721726027529
        norm:
          error: 1.2070508525157752e-08
          unit: 1 / (m2 s TeV)
          value: 2.3400361534892314e-07
        reference:
          error: 0.0
          unit: TeV
          value: 0.9999999999999999
      spectral_model: PowerLaw

Running individual steps
------------------------

You can also run the individual steps of the analysis

.. code-block:: bash

    $ gammapy-spectrum --help
    Usage: gammapy-spectrum [OPTIONS] COMMAND [ARGS]...

    Gammapy tool for spectrum extraction and fitting.

      Examples
      --------

      gammapy-spectrum extract config.yaml
      gammapy-spectrum fit config.yaml
      gammapy-spectrum all config.yaml

    Options:
       -h, --help  Show this message and exit.

    Commands:
      all      Fit spectral model to 1D spectrum
      display  Display results of spectrum fit
      extract  Extract 1D spectral information from event...
      fit      Fit spectral model to 1D spectrum
      plot     Plot spectrum results file




