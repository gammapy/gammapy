.. _dataformats_observation_lists:

Observation lists
=================

Run lists specify a list of observation runs to be processed by a given tool.

In the end the tool will usually generate input and output data filenames from
the information in the run list. These filenames can either be given directly
or there can be naming conventions, e.g. the event list for run ``42`` could be stored
at ``$TEV_DATA/RUN_000042/Events.fits``.


CSV format
----------

We use the `CSV <http://en.wikipedia.org/wiki/Comma-separated_values>`_ (comma-separated values) format for run lists.
This has the advantage that everyone can work with whatever tool they like. Here's some good options:

* `LibreOffice Calc <http://www.libreoffice.org/discover/calc/>`_ 
* `TOPCAT <http://www.star.bris.ac.uk/~mbt/topcat/>`_ or `STILTS <http://www.star.bris.ac.uk/~mbt/stilts/>`_
* `csvkit <https://csvkit.readthedocs.org/en/latest/>`_
* Your own script using e.g the `Python standard library CSV module <http://docs.python.org/2/library/csv.html>`_ or `pandas <http://pandas.pydata.org>`_

A run list must have at least a column called ``OBS_ID``::
 
   OBS_ID
   42
   43

Usually it has many more columns with information about each observation. A list of Gammapy supported columns is::

   OBS_ID: observation ID as an integer (starting at 0 or 1? does it matter?)
   RA: pointing position right ascension in equatorial (ICRS) coordinates
   DEC: pointing position declination in equatorial (ICRS) coordinates
   AZ: average azimuth angle during the observarion
   ALT: average altitude angle during the observarion
   MUON_EFFICIENCY: average muon efficiency of the telescopes
   TIME_START: start time of the observation stored as number of seconds after the time reference in the header
   TIME_STOP: end time of the observation in the same forma as TIME_START
   TIME_OBSERVATION: duration of the observation
   TIME_LIVE: duration of the observation without dead time
   TRIGGER_RATE: average trigger rate of the system
   MEAN_TEMPERATURE: average temperature of the atmosphere
   N_TELS: number of telescopes participating in the observation
   TEL_LIST: string with a CSV list of telescope IDs participating in the observation
   QUALITY: data quality; recommended: "spectral" or "detection" (not used by Gammapy at the moment)

Extra user defined columns are allowed; Gammapy will ignore them.

In order for the extra columns to have full meaning a header is needed with at least the following keywords::

   OBSERVATORY_NAME: name of the observatory where the observations were taken. This is important for instance for coordinate transformations between celestial (i.e. RA/dec) and terrestrial (i.e. az/alt) coordinate systems.
   TIME_REF_MJD_INT: reference time for other times in the list (i.e. TIME_START/TIME_STOP). Integer value in mean julian days.
   TIME_REF_MJD_FRA: fraction of integer value defined in TIME_REF_MJD_INT.

TODO: change names already defined in event lists (and in `~gammapy.utils.time.time_ref_from_dict`) ``MJDREFI`` and ``MJDREFF``? Or adopt ``TIME_REF_MJD_INT`` and ``TIME_REF_MJD_FRA``?

TODO: should the observation list have already a header? Or should the header values be defined as columns, then interpreted correspondingly when stored into an ObservationTable? (I might be mixing up 2 things: obs lists and obs tables here? or should they have the same format?)

TODO: should the observation list already define the times in this ``TIME_REF_MJD_INT + TIME_REF_MJD_FRA`` format? Or should it be converted once its read into an ObservationTable?

TODO: ``TEL_LIST``: incompatibility wit CSV format, since it's a CSV list of tels???!!!

TODO: how to biuld sphinx only for a specific module or file?


XML format
----------

GammaLib / ctools uses an "observation definition" XML format described
`here <http://gammalib.sourceforge.net/user_manual/modules/obs.html#describing-observations-using-xml>`__.
