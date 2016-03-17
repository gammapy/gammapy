.. include:: ../references.txt

.. _obs_dm:

Data Management
===============

Classes
-------

Gammapy helps with data management using a multi-layered set of classes. The job of the DataManager and DataStore is to
make it easy and fast to locate files and select subsets of observations.

* The `~gammapy.data.DataManager` represents a configuration (usually read
  from a YAML file) of directories and index files specifying where
  data is available locally and remotely and in which formats.
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


Data Manager
++++++++++++

The data access is even more convenient with a DataManager.It is based one a data registry config file (YAML format)
that specifies where data and index files are located on the user's machine. In other words, the data registry is
a list of datastores that can be accessed by name. By default, Gammapy looks for data registry config files called
``data-register.yaml`` in the ``~/.gammapy`` folder. Thus, put the following in ``~/.gammapy/data-register.yaml``
in order to proceed with the example.

.. include:: ./example-data-register.yaml
    :code: yaml


Now the data access work like this

.. code-block:: python

    >>> from gammapy.data import DataManager
    >>> data_manager = DataManager.from_yaml(DataManager.DEFAULT_CONFIG_FILE)
    >>> data_manager.store_names
    ['crab_example']
    >>> data_store = data_manager.stores[0]

or just

.. code-block:: python

    >>> from gammapy.data import DataStore
    >>> data_store = DataStore.from_name('crab_example')


Command line tools
------------------

* ``gammapy-data-manage`` -- Manage data locally and on servers
* ``gammapy-data-browse`` -- A web app to browse local data (stats and quick look plots)


.. _dm_formats:

Data formats
------------

Data registry config file
+++++++++++++++++++++++++

Here's an example of a data store registry YAML config file that Gammapy understands:

.. include:: ../../gammapy/data/tests/data/data-register.yaml
    :code: yaml

Observation table
+++++++++++++++++

The required format of observation lists is described at :ref:`dataformats_observation_lists`.

File table
++++++++++

File tables should contain the following columns (not all required)

============  ============================================================================
Name          Description
============  ============================================================================
OBS_ID        Observation ID (int)
TYPE          File type (str)
NAME          File location relative to ``BASE_DIR`` (str)
SIZE          File size in bytes
MTIME         File modification time (double)
MD5           MD5 checksum (str)
HDUNAME       HDU extension name (str)
HDUCLASS      HDU class (str)
============  ============================================================================

* ``TYPE`` should be one of the formats listed at :ref:`dataformats_overview`.
* ``HDUCLASS`` should be one of the formats listed at :ref:`dataformats_overview`.
* ``NAME`` are relative to a ``BASE_DIR`` that is not stored in the file table,
  but must be supplied by the user in the data registry config file
  (it will change if data is copied to other machines, and we want to keep the file index
  table unchanged).

The file table header can contain information such as:

============  ============================================================================
Name          Description
============  ============================================================================
TELESCOP      Mission name
CHAIN         Analysis chain
CONFIG        Analysis configuration
...           ...
============  ============================================================================



