.. include:: ../references.txt

.. _overview_DL3:

The Data Level 3 (DL3) format
=============================

The data level 3 FITS files consisting of event lists and extra information concerning the observation
(pointing direction, time), as well as two index tables that list the
observations and declare which response should be used with which event data.

Data levels in CTA
------------------

General scheme
^^^^^^^^^^^^^^

Add a figure here.


:math:`\gamma`-like events
^^^^^^^^^^^^^^^^^^^^^^^^^^

Instrument response functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each observation, instrument response functions (namely effective area, point spread function, energy
dispersion, background) are distributed. Some details about the origin of these functions are given
in :ref:`irf-theory`. The functions are stored in the form of multidimensional tables giving the IRF
value as a function of position in the field-of-view and energy of the incident photon.

The formats used are discussed and described in `gadf`_. This format is still a prototype. In the coming
years CTA will develop and define it's release data format, and Gammapy  will adapt to that.


Application in gammapy
----------------------

The main classes in Gammapy to access the DL3 data library are the
`~gammapy.data.DataStore` and `~gammapy.data.Observation`.
They are used to store and retrieve dynamically the datasets
relevant to any observation (event list in the form of an `~gammapy.data.EventList`,
IRFs see :ref:`irf` and other relevant informations).

Once some observation selection has been selected, the user can build a list of observations:
a `~gammapy.data.Observations` object, which will be used for the data reduction process.
A convenient way to do this is to use the high level interface, see :ref:`gammapy.analysis <analysis>`.

Example notebooks
^^^^^^^^^^^^^^^^^

* `CTA DL3 handling <../tutorials/cta.html>`__
* `HESS Data Release 1 handling <../tutorials/hess.html>`__

Relevant API
^^^^^^^^^^^^

* :ref:`gammapy.data API documentation<data>`
* :ref:`gammapy.irf API documentation<irf>`

