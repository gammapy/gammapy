.. include:: ../../references.txt

.. _tutorials-gammapy-spectrum:

Spectral fitting with ``gammapy-spectrum``
==========================================

In this tutorial you will perform a spectra fit starting from an
`~gammapy.data.EventList` and a set of 2D IRFs
(`~gammapy.irf.EnergyDispersion2D`, `~gammapy.irf.EffectiveAreaTable2D`).
The data used in this tutorial consist of 2 simulated
Crab observations with the H.E.S.S. array. They are available in gammapy-extra.

.. _tutorials-gammapy-spectrum-extract:

Extracting 1D spectral information
----------------------------------

The first step on our way to the Crab spectrum is generating 1D spectral
information, i.e. an on `~gammapy.spectrum.CountsSpectrum`, and off
`~gammapy.spectrum.CountsSpectrum`, an `~gammapy.irf.EffectiveAreaTable`
vector, and an `~gammapy.irf.EnergyDispersion` matrix. This is done with
``gammapy-spectrum extract``. Below you find an example config file:


.. literalinclude:: ./spectrum_extraction_example.yaml
    :language: yaml
    :linenos:


In Detail:

* Line 4-7 : Specify the `~gammapy.data.DataStore` to take the data from,
  a list of observations (list or filename),
  and the number of observations to analyse (0: all observations)
* Line 9-13 : Define reconstructed energy binning, the true energy binning is
  derived from the IRF files.
* Line 15-20 : Define region to extract the spectrum from (on region)
* Line 22-25 : Choose background estimation method.
  For available methods see :ref:`spectrum_background_method`
* Line 27-28 : Specify fits map containing exclusion regions. Here a map
  excluding all `TevCat`_ sources is used.
* Line 30-33 : Define output folder and files.

To run the spectral extraction copy the above config file to your machine,
to e.g. ``crab_config.yaml`` and run

.. code-block:: bash

   gammapy-spectrum extract crab_config.yaml

This creates the folder ``crab_analysis`` in you current working directory. In
this folder you find the generated OGIP data in the folder ``ogip_data``.
A detailed description of the OGIP files, can be found under :ref:`gadf:ogip`.
The ``spectrum_stats.yaml`` file holds all results of the spectrum extration step.
You can examine this file as explained in :ref:`tutorials-gammapy-spectrum-examine`.
Furthermore, there is an ``observations.fits`` file holding an `~gammapy.data.ObservationTable`. It
points to the OGIP data for each run, which serves as input for ``gammapy-spectrum fit``.
Note that at the moment it is necessary to create OGIP files at the spectrum
extraction step, the data cannot be transferred in-memory to ``gammapy-spectrum fit``

``gammapy-spectrum extract --interactive`` will drop you into an IPython session
  after the extraction step, where you can interactively look at all analysis
  results.

If you do not want to actually create all the 1D spectrum objects but only
have a look at analysis parameters like the offset distribution you can run
``gammapy-spectrum extract --interactive --dry-run``.

.. _tutorials-gammapy-spectrum-fit:

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

This creates the file ``fitresult.yaml`` in the ``crab_analysis`` folder, which
can be used as shown in :ref:`tutorials-gammapy-spectrum-examine`.

``gammapy-spectrum fit`` also has an ``--interactive`` flag.

If you do not care about intermediate analysis result you can use
``gammapy-spectrum all`` to run the extraction and the fitting step in one go.

.. _tutorials-gammapy-spectrum-examine:

Examining all results
---------------------

``gammapy-spectrum`` features two tools to examine and compare fit results.

* ``gammapy-spectrum display`` takes any number of results files as arguments
  and prints a comparison table. :download:`This <./spectrum_stats_23592.yaml>`
  file contains the `~gammapy.spectrum.results.SpectrumStats` for a Crab analysis
  using only one of the runs. Copy it to your analysis directory to run the
  following command

.. code-block:: bash

    gammapy-spectrum display crab_analysis/spectrum_stats.yaml spectrum_stats_23592.yaml
                     analysis             n_off n_on excess energy_range [2] alpha
                                                              TeV
    --------------------------------- ----- ---- ------ ---------------- ------
    crab_analysis/spectrum_stats.yaml   414  416    354      0.01 .. 300  0.149
            spectrum_stats_23592.yaml   252  197    176      0.01 .. 300 0.0833

If you want to customize this output table you could for example run

.. code-block:: bash

    gammapy-spectrum display crab_analysis/spectrum_stats.yaml spectrum_stats_23592.yaml --cols analysis,n_on,alpha --sort n_on --identifiers two_runs,one_run
    analysis n_on alpha

    -------- ---- ------
     one_run  197 0.0833
    two_runs  416  0.149

The ``--browser`` flag lets you examine your comparison table in the browser

* ``gammapy-spectrum plot`` can be used to plot fit results.

TODO




