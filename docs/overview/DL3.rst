.. _DL3_general:

The Data Level 3 (DL3) format
=============================

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
relevant to any observation (event list, IRFs and other relevant informations).


`~gammapy.data.Observations`, `~gammapy.data.EventList` and various other
classes. To learn how to work with this data, and how to reduce it to the
datasets level, see :ref:`gammapy.analysis <analysis>`, :ref:`gammapy.data
<data>`, :ref:`gammapy.irf <irf>`, :ref:`gammapy.cube <cube>`,
:ref:`gammapy.spectrum <spectrum>`.

Example notebooks
^^^^^^^^^^^^^^^^^

Relevant API
^^^^^^^^^^^^
