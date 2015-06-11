.. _dataformats:

************
Data Formats
************

Here we describe various data file formats that are useful for the exchange
of TeV data, instrument response functions and results.

.. note:: In :ref:`datasets` you find example gamma-ray datasets,
          some in the formats described here.

Where available and useful existing standards are used, e.g.
for spectral data the X-ray community has developed the ``PHA``, ``ARF`` and ``RMF``
file formats and they have developed powerful tools to work with data in that format.

In other cases (flux points, point spread function, run lists) there is no standard and
we simply use general-purpose file formats define our own semantics as described
in the :ref:`dataformats_file_formats` section. 


.. _time_gammapy:

***************
Time in Gammapy
***************

Gammapy adopts the exact same time standard as the Fermi Science Tools as described on this page: `Cicerone: Data - Time in Fermi Data Analysis <http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/Time_in_ScienceTools.html>`_.
Every time should be defined as a `~astropy.time.Time` object, representing UTC seconds after the "Mission elapsed time" [MET]_ origin. Each experiment may have a different [MET]_ origin that should be included in the header of the data files.


.. toctree::
   :maxdepth: 1

   file_formats

   events
   maps
   cubes
   spectra
   lightcurves

   psf
   observation_lists
   source_models
   target_lists
