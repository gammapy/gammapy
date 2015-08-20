.. _background_make_models:

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

* ``bg_observation_table.fits.gz``: total observatin table used for
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
  counts (events) and a livetime correction data cubes.

.. _background_make_models_datasets_for_testing:

Datasets for testing
--------------------

In order to test the background generation tools, either real data
from an existing experiment such as H.E.S.S. can be used. Since the
data of current experiments is not public. Rudimentary tools to
prepare a dummy dataset have been placed in Gammapy:

* `~gammapy.datasets.make_test_dataset`: function to produce a dummy
  observation list and its corresponding dataset
  consisting on event lists and effective area tables and store
  everything on disk.

* `~gammapy.datasets.make_test_eventlist`: function called
  recursivelly by `~gammapy.datasets.make_test_dataset` to produce
  the data (event lists and effective area table)corresponding to
  one observation.

They are very easy to use. A H.E.S.S.-like test dataset can be
produced with a few lines of python code:

.. code-block:: python

    fits_path = '/path/to/fits/event_lists/base/dir'
    observatory_name = 'HESS'
    n_obs = 2
    random_state = np.random.RandomState(seed=0)

    make_test_dataset(fits_path=fits_path,
                      observatory_name=observatory_name,
                      n_obs=n_obs,
                      random_state=random_state)

Then the data can be read back using the `~gammapy.obs.DataStore`
class, and eg. print the observation table and the names of the filescreated with a few extra lines of python code:

.. code-block:: python

    scheme = 'HESS'
    data_store = DataStore(dir=fits_path, scheme=scheme)
    observation_table = data_store.make_observation_table()
    print(observation_table)
    event_list_files = data_store.make_table_of_files(observation_table,
                                                      filetypes=['events'])
    aeff_table_files = data_store.make_table_of_files(observation_table,
                                                      filetypes=['effective area'])
    for i_ev_file, i_aeff_file in zip(event_list_files['filename'],
                                      aeff_table_files['filename']):
        print(' ev infile: {}'.format(i_ev_file))
        print(' aeff infile: {}'.format(i_aeff_file))
