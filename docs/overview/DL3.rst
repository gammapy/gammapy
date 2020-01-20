.. _overview_DL3:

The Data Level 3 (DL3) format
=============================

The data level 3  FITS files consisting of event lists,
instrument response information (effective area, point spread function, energy
dispersion, background) and extra information concerning the observation
(pointing direction, time), as well as two index tables that list the
observations and declare which response should be used with which event data

Data levels in CTA
------------------

General scheme
^^^^^^^^^^^^^^

Add a figure here.


:math:`\gamma`-like events
^^^^^^^^^^^^^^^^^^^^^^^^^^

Instrument response functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note: the current IACT DL3 data model
and format is a prototype (documented at `gadf`_), in the coming years CTA will
develop and define it's release data format, and Gammapy and other IACTs will
adapt to that.

Application in gammapy
----------------------

The main classes in Gammapy to access the DL3 data library are the
`~gammapy.data.DataStore` and `~gammapy.data.DataStoreObservation`.
They are used to store and retrieve dynamically the datasets
relevant to any observation (event list in the form of an `~gammapy.data.EventList`,
IRFs see :ref:`irf` and other relevant informations).

Once some observation selection has been selected, the user can build a list of observations:
a `~gammapy.data.Observations` object, which will be used for the data reduction process.
A convenient way to do this is to use the high level interface, see :ref:`gammapy.analysis <analysis>`.

Example notebooks
^^^^^^^^^^^^^^^^^

* CTA DL3 handling <notebooks/cta.html>`__
* HESS Data Release 1 handling <notebooks/hess.html>`__

Relevant API
^^^^^^^^^^^^

* :ref:`gammapy.data<data API documentation>`
* :ref:`gammapy.irf <irf API documentation>`

