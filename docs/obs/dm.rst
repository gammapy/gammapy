.. include:: ../references.txt

.. _obs_dm:

Data Management
===============

Classes
-------

Gammapy helps with data management using a multi-layered set of classes:

* The `~gammapy.obs.DataManager` represents a configuration (usually read
  from a YAML file) of directories and index files specifying where
  data is available locally and remotely and in which formats.
* The `~gammapy.obs.DataStore` represents data files in a given directory
  and usually consists of two things: a `~astropy.table.Table`
  that contains the location, content, size, checksum of all files
  and a `~gammapy.obs.ObservationTable` that contains relevant parameters
  for each observation (e.g. time, pointing position, ...)
* The actual data and IRFs are represented by classes,
  e.g. `~gammapy.data.EventList` or `~gammapy.irf.EffectiveAreaTable2D`, ...

The job of the DataManager and DataStore is to make it easy and fast
to locate files and select subsets of observations.

.. code-block:: python

    >>> from gammapy.obs import DataManager
    >>> data_manager = DataManager.from_yaml('data-register.yaml')
    >>> data_store = data_manager['hess-hap-prod01']
    >>> data_store.filename(obs_id=89565, filetype='AEFF')
    /Users/deil/work/_Data/hess/fits/hap-hd/fits_prod01/std_zeta_fullEnclosure/run089400-089599/run089565/hess_aeff_2d_089565.fits.gz


In addition, these classes contain convenience properties and methods that
actually load the data and IRFs and return objects of the appropriate class

.. code-block:: python

    >>> aeff = data_store.load(obs_id=89565, filetype='AEFF')
    >>> aeff.__class__.name
    TODO: gammapy.irf.EffectiveAreaTable2D


Command line tools
------------------

* ``gammapy-data-manage`` -- Manage data locally and on servers
* ``gammapy-data-browse`` -- A web app to browse local data (stats and quick look plots)


Data formats
------------

Users should have one data registry config file (YAML format) that specifies where data
and index files are located on the user's machine.

The data registry is basically a list of datastores that can be accessed by name.

Each data store has two index files:

* Observation table (FITS format)
* File table (FITS format)

Data registry config file
+++++++++++++++++++++++++

Here's an example of a data store registry YAML config file that Gammapy understands:

.. include:: ../../gammapy/obs/tests/data/data-register.yaml
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



