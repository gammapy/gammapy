.. include:: ../references.txt

.. _obs_dm:

Data Management
===============

Classes
-------

Gammapy helps with data management using a multi-layered set of classes. The job of the DataStore is to
make it easy and fast to locate files and select subsets of observations.

* The `~gammapy.data.DataStore` represents data files in a given directory
  and usually consists of two things: a `~astropy.table.Table`
  that contains the location, content, size, checksum of all files
  and a `~gammapy.data.ObservationTable` that contains relevant parameters
  for each observation (e.g. time, pointing position, ...)
* The actual data and IRFs are represented by classes,
  e.g. `~gammapy.data.EventList` or `~gammapy.irf.EffectiveAreaTable2D`.

Getting Started
---------------

The following example demonstrates how data management is done in Gammapy. It uses a test data set, which is available
in the `gammapy-extra <https://github.com/gammapy/gammapy-extra>`__ repository. Please clone this repository and
navigate to ``gammapy-extra/datasets/``. The folder ``hess-crab4-hd-hap-prod2`` contains IRFs and simulated event lists for 4
observations of the Crab nebula. It also contains two index files:

* Observation table `observations.fits.gz`
* File table `files.fits.gz`

These files tell gammapy which observations are contained in the data set and where the event list and IRF files are
located for each observation (for more information see :ref:`dm_formats`).

.. _data_store:

Data Store
++++++++++

Exploring the data using the DataStore class works like this


.. code-block:: python

    >>> from gammapy.data import DataStore
    >>> data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')
    >>> data_store.info()
    Data store summary info:
    name: noname
    base_dir: hess-crab4
    observations: 4
    files: 16
    >>> data_store.obs(obs_id=23592).location(hdu_class='events').path(abs_path=False)
    'hess-crab4/hess_events_simulated_023592.fits'

In addition, the DataStore class has convenience properties and methods that
actually load the data and IRFs and return objects of the appropriate class

.. code-block:: python

    >>> event_list = data_store.obs(obs_id=23592).events
    >>> type(event_list)
    TODO
    >>> aeff2d = data_store.obs(obs_id=23592).aeff
    >>> type(aeff2d)
    <class 'gammapy.irf.effective_area_table.EffectiveAreaTable2D'>
    >>> obs.target_radec
    <SkyCoord (FK5: equinox=J2000.000): (ra, dec) in deg
	(83.63333333, 22.01444444)>


.. _dm_formats:

Data formats
------------

See :ref:`gadf:iact-storage`.
