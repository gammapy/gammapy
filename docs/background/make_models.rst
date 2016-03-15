.. _background_make_background_models:

Make background models
======================

Gammapy tools to produce background models.

Make cube background models
---------------------------

The ``gammapy-make-bg-cube-models`` command line tool can be used to produce
background cube models from the data files at a given path location.

For more details, please refer to `~gammapy.scripts.make_bg_cube_models`.

Examples
~~~~~~~~

The ``gammapy-make-bg-cube-models`` tool has a few options. For a full list of options, please use:

.. code-block:: bash

    $ gammapy-make-bg-cube-models --help

at the command line.

Command examples:

* Create background models using a H.E.S.S. or H.E.S.S.-like dataset
  (can take a few minutes):

  .. code-block:: bash

      $ gammapy-make-bg-cube-models /path/to/fits/event_lists/base/dir \
                                    HESS bg_cube_models

* Run a quick test using only a few runs:

  .. code-block:: bash

      $ gammapy-make-bg-cube-models /path/to/fits/event_lists/base/dir \
                                    HESS bg_cube_models --test

* Create background models using the method developped in
  [Mayer2015]_ (almost equal to the default case for now):

  .. code-block:: bash

      $ gammapy-make-bg-cube-models /path/to/fits/event_lists/base/dir \
                                    HESS bg_cube_models --method michi

The output files are created in the output directory:

* ``bg_observation_table.fits.gz``: total observation table used for
  the models. The list has been filtered from observations taken at
  or nearby known sources.

* ``bg_observation_table_grouped.fits.gz``: observation table grouped
  according to the selected binning for the background observations.

* ``bg_observation_groups.ecsv``: table describing the observation
  groups.

* ``bg_cube_model_group<ID>_<format>.fits.gz``: files containing the
  background models for each group **ID** in 2 different **format**
  kinds: *table*, for data analysis and *image* for a quick
  visualization using eg. ``DS9``. The table files contain also a
  counts (a.k.a. events) and a livetime correction data cubes.

In order to compare 2 sets of background cube models,
the ``examples/wip_bg_cube_model_comparison.py`` can be used.

.. _background_make_background_models_datasets_for_testing:

Datasets for testing
~~~~~~~~~~~~~~~~~~~~

In order to test the background model generation tools, real
data from existing experiments such as H.E.S.S. can be used.
Since the data of current experiments is not public, tools to
simulate datasets are described in :ref:`datasets_obssim`.

There is also a tool in Gammapy to simulate background cube models:
`~gammapy.datasets.make_test_bg_cube_model`.
It can be used to produce true background cube models to use to
compare to the reconstructed ones produced with the machinery
described above, using a simulated dataset using the tools from
:ref:`datasets_obssim`. If using the same model
for producing the simulated dataset and the true background cube
models, the reconstructed ones produced with
``gammapy-make-bg-cube-models`` should match the true ones.

The example script :download:`wip_bg_cube_models_true_reco.py
<../../examples/wip_bg_cube_models_true_reco.py>` can be used
to produce a true cube bg model and a reco cube bg model using the
same model (except for absolute normalization). The models can be
used to test the cube bg model production and can be compared to each
other using the :download:`wip_bg_cube_model_comparison.py
<../../examples/wip_bg_cube_model_comparison.py>` example script.

Comparing true-reco models
**************************

Two model files located in the ``gammapy-extra`` repository have been
produced using the example script :download:`wip_bg_cube_models_true_reco.py
<../../examples/wip_bg_cube_models_true_reco.py>`:

* `bg_cube_model_true.fits.gz
  <https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/background/bg_cube_model_true.fits.gz>`_
  is a true bg cube model produced with
  `~gammapy.datasets.make_test_bg_cube_model`.
* `bg_cube_model_reco.fits.gz
  <https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/background/bg_cube_model_reco.fits.gz>`_
  is a reco bg cube model produced with
  `~gammapy.background.make_bg_cube_model`, using dummy data produced
  with `~gammapy.datasets.make_test_dataset`.

The following plots are produced with a modified version of the
:download:`wip_bg_cube_model_comparison.py
<../../examples/wip_bg_cube_model_comparison.py>` example script:

TODO: remove or fix these examples

.. .. plot:: background/plot_bgcube_true_reco.py

The input counts spectrum is a power-law with an index of 1.5, in
order to have some counts at high energies with a reasonable amount
of simulated data. In reality the background spectrum has a spectral
index close to 2.7.

The bg rate appears as a spectrum of **index + 1** (2.5 in this
example).
The reason being that, in order to produce the bg model, the
contents of the cube (counts per unit time) have to be divided by the
bin volume (delta x * delta y * delta E). When computing
counts/delta E, the index of the bg rate increases by 1 w.r.t. the
index of the power-law spectrum used to sample (or model) the counts.
