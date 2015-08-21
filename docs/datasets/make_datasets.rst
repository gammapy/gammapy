.. _datasets_make_datasets_for_testing:

Make datasets for testing
=========================

In order to test some of the tools from Gammapy, such as the
background generation tools
(:ref:`background_make_background_models`), real data
from existing experiments such as H.E.S.S. can be used. Since the
data of current experiments is not public, rudimentary tools to
prepare a dummy dataset have been placed in Gammapy:

* `~gammapy.datasets.make_test_dataset`: function to produce a dummy
  observation list and its corresponding dataset
  consisting on event lists and effective area tables and store
  everything on disk.

* `~gammapy.datasets.make_test_eventlist`: function called
  recursivelly by `~gammapy.datasets.make_test_dataset` to produce
  the data (event lists and effective area table) corresponding to
  one observation.

Currently only background events are simulated (no signal),
following a very simple model, and only a few
columns of the `~gammapy.data.EventList` class are filled. In
addition, the effective area files produced, are empty except fot
the low energy threshold header entry.

The tools are very easy to use. A H.E.S.S.-like test dataset can be
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
