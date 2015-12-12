.. _dataformats_observation_lists:

Observation lists
=================

Observation lists specify a list of observations (a.k.a. runs) and are
represented in Gammapy as `~gammapy.data.ObservationTable`, which is a sub-class
of `~astropy.table.Table`.

Because there is no standard defined for observation lists, everyone uses their own format.
E.g. one person calls the pointing position columns ``RA_PNT`` and ``DEC_PNT``, another
calls them ``RA`` and ``DEC``.

In Gammapy we only support one format, all Gammapy-internal code assumes
the column names and units shown below.
Some of these names are cryptically short, to satisfy the use case where
people want to store these parameters in FITS headers, which have an 8-character
limit on the keys.

For convenience, we provide a few helper functions and arguments to convert other formats
into this format on read. If that doesn't work for your lists, you have to reformat
it yourself.

Note that all of these columns are optional, i.e. a table that just contains an
``OBS_ID`` column is a valid observation table and is enough to look up data
files via the `~gammapy.data.DataStore` or `~gammapy.data.DataManager` classes.
Other observation lists might only contain ``RA`` and ``DEC`` and some processing might
not require the presence of the ``OBS_ID`` or any other columns.
To accommodate these various use cases for observation lists and to avoid forcing
users to add extra columns they don't really need, we don't check for the presence of
a given list of columns on read, but just access columns in the computations where needed.
This implies that Gammapy can fail late in scripts with a ``KeyError``, and you'll just
have to fix the format and re-run.


============  ============================================================================
column name   description
============  ============================================================================
OBS_ID        Observation ID (int)
RA_PNT        Right ascension of pointing position (float, degree)
DEC_PNT       Declination of pointing positoin (float, degree)
AZ_PNT        Azimuth at observation mid-time (float, degree)
ALT_PNT       Altitude at observation mid-time (float, degree)
TSTART        Observation start time in MET (float, seconds)
TSTOP         Observation end time in MET (float, seconds)
ONTIME        Observation duration = TSTOP - TSTART (float, seconds)
LIVETIME      Observation live time (duration minus dead time) (float, seconds)
MEANTEMP      Temperature at observation mid-time (float, Celsius)
N_TELS        Number of telescopes participating in the observation (int)
TELLIST       Comma-separated list of telescope IDs participating in the observation (str)
MUONEFF       Mean muon efficiency of the telescopes
QUALITY       Data quality (int)
============  ============================================================================

Notes:

* Longitude angles such as right ascension, Galactic longitude, or azimuth should
  be wrapped at **360 deg**, in other words, they should be defined in the
  ``[0, 360) deg`` interval.
* Data quality: TODO

In order for the extra columns to have full meaning the following is needed:

 * Extra row right after the column name, specifying the unit of the quantity listed on each column.
 * A header with at least the following keywords:

================  ============================================================================  =========
keyword           description                                                                   required?
================  ============================================================================  =========
OBSERVATORY_NAME  name of the observatory where the observations were taken; this is            no
                  important for instance for coordinate transformations between celestial
                  (i.e. RA/dec) and terrestrial (i.e. az/alt) coordinate systems
MJDREFI           reference time: integer value in mean julian days; details in                 yes?
                  :ref:`time_handling`
MJDREFF           reference time: fraction of integer value defined in **MJDREFI**; details     yes?
                  in :ref:`time_handling`
TIME_FORMAT       format in which times are stored: *absolute* (UTC) or *relative* ([MET]_);    yes?
                  see details for both formats in :ref:`time_handling`
================  ============================================================================  =========
