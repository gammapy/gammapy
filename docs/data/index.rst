.. include:: ../references.txt

.. _data:

*****************************
Data classes (`gammapy.data`)
*****************************

.. currentmodule:: gammapy.data

Introduction
============

`gammapy.data` contains classes to represent gamma-ray data.

We follow the Fermi data model and FITS formats as much as possible.

The data format used is FITS and we define one container class to represent
each FITS extension. In addition we define two high-level dataset classes
that group all info data and metadata that's usually given for a single
observation together:

* Unbinned data is represented by a `~gammapy.data.EventListDataset`, which contains:

  - `~gammapy.data.EventList` - table with time, position and energy for each event
  - `~gammapy.data.GoodTimeIntervals` - table of good time intervals.
    Used for livetime and exposure computation.
  - `~gammapy.data.TelescopeArray` - some info about the array that took the data.
    Optional, not used at the moment.

* For most analysis the first step is to bin the data, which turns the
  `~gammapy.data.EventListDataset` into one of:

  - `~gammapy.data.CountsCubeDataset` (lon, lat, energy)
  - `~gammapy.data.CountsSpectrumDataset` (energy)
  - `~gammapy.data.CountsImageDataset` (lon, lat)
  - `~gammapy.data.CountsLightCurveDataset` (time)

* TODO: add IRFs to the dataset classes?

* TODO: do we need the ``*Dataset`` wrappers or can we only have
  ``EventList``, ``CountsCube``, ``CountsSpectrum``, ``CountsImage``, ...?

* We'll have to see if we want to copy the IRF and / or GTI and / or TelescopeArray info
  over to the binned dataset classes ... at the moment it's not clear if we need that info.

Energy binning
--------------

* `~gammapy.spectrum.EnergyBinCenters` and FITS "ENERGIES" extensions.
* `~gammapy.spectrum.EnergyBinEdges` and FITS "EBOUNDS" extensions.

Spatial binning
---------------

TODO: Should we define a "SpatialGrid" or "SpatialBinning" class that wraps
the 2d image FITS WCS and add convenience methods like generating them from scratch
e.g. centered on the target or aligned to a survey map grid?

Getting Started
===============


.. code-block:: python

    >>> from gammapy.data import EventListDataset
    >>> events_ds = EventListDataset.read('events.fits')
    >>> events_ds.info()
    >>> from gammapy.data import CountsCubeDataset
    # This is just an idea ... not implemented!
    >>> counts_ds = CountsCubeDataset.from_events(events_ds, ...)
    >>> counts_ds = events_ds.make_counts_cube(...)

Reference/API
=============

.. automodapi:: gammapy.data
    :no-inheritance-diagram:
