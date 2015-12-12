.. include:: ../references.txt

.. _datasets_obssim:

Simulate event lists
====================

Here we describe how to simulate event lists for a given
observation list, source model and instrument.

Gammapy is mostly an analysis package for binned analysis,
and so far we haven't implemented general tools to sample
arbitrary spatial and spectral density distributions.

An excellent tool `ctobssim`_ is available within the ctools
package, so here we'll describe how to use that first and then
mention the existing functionality in Gammapy.

.. _datasets_obssim_ctobssim:

Using ctobssim
++++++++++++++

Using HESS IRFs and a real event list as input (for header info),
it's possible to simulate event lists according to any source and background model you like.

See https://github.com/gammapy/gammapy-extra/blob/master/datasets/hess_crab/make2.py
as an example, which was used to generated the
https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/irf/hess/pa/hess_events_023523.fits.gz
example file.

TODO: we should extend this to a Gammapy command line tool that:

- can simulate simple IRF files for HESS or CTA (so that we don't have to use real HESS IRFs)
- can process an observation list, i.e. it's not necessary to have real observations.

For now this one simulated HESS observation can be used for testing in Gammapy.

.. _datasets_obssim_gammapy:

Using Gammapy
+++++++++++++

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
produced with a few lines of Python code:

.. code-block:: python

    workdir = gammapy/scripts/tests/test_make_bg_cube_models.py
    make_test_dataset(outdir=workdir,
                      observatory_name='HESS,
                      n_obs=2,
                      random_state=0)

Then the data can be read back using the `~gammapy.data.DataStore`
class, and eg. print the observation table and the names of the files
created with a few extra lines of Python code:

.. code-block:: python

    data_store = DataStore.from_dir(dir=workdir)
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
