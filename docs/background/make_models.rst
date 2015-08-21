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
                                    HESS bg_cube_models --a-la-michi

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

In order to compare 2 sets of background cube models, the following
script in the `examples` directory can be used:
:download:`plot_bg_cube_model_comparison.py
<../../examples/plot_bg_cube_model_comparison.py>`

Datasets for testing
~~~~~~~~~~~~~~~~~~~~

In order to test the background model generation tools, real
data from existing experiments such as H.E.S.S. can be used. Since
the data of current experiments is not public, rudimentary tools to
prepare a dummy dataset have been placed in Gammapy:
:ref:`datasets_make_datasets_for_testing`.
