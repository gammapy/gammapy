.. _dataformats_observation_lists:

Observation lists
=================

Run lists specify a list of observation runs to be processed by a given tool.

In the end the tool will usually generate input and output data filenames from
the information in the run list. These filenames can either be given directly
or there can be naming conventions, e.g. the event list for run ``42`` could be stored
at ``$TEV_DATA/RUN_000042/Events.fits``.

A run list must have at least a column called ``OBS_ID``::
 
   OBS_ID
   42
   43

Usually it has many more columns with information about each observation. A list of Gammapy supported columns is:

================  ================================================================================================  =========
column name          description                                                                                       required?
================  ================================================================================================  =========
OBS_ID            observation ID as an integer                                                                      yes
RA                pointing position right ascension in equatorial (ICRS) coordinates                                yes?
DEC               pointing position declination in equatorial (ICRS) coordinates                                    yes?
AZ                average azimuth angle during the observation                                                      yes?
ALT               average altitude angle during the observation                                                     yes?
MUON_EFFICIENCY   average muon efficiency of the telescopes                                                         yes?
TIME_START        start time of the observation stored as number of seconds after the time reference in the header  yes?
TIME_STOP         end time of the observation in the same forma as TIME_START                                       yes?
TIME_OBSERVATION  duration of the observation                                                                       yes?
TIME_LIVE         duration of the observation without dead time                                                     yes?
TRIGGER_RATE      average trigger rate of the system                                                                no
MEAN_TEMPERATURE  average temperature of the atmosphere                                                             no
N_TELS            number of telescopes participating in the observation                                             yes?
TEL_LIST          string with a CSV list of telescope IDs participating in the observation                          yes?
QUALITY           data quality; recommended: "spectral" or "detection" (not used by Gammapy at the moment)          no
================  ================================================================================================  =========

Extra user defined columns are allowed; Gammapy will ignore them.

In order for the extra columns to have full meaning the following is needed:

 * Extra row right after the column name, specifying the unit of the quantity listed on each column.
 * A header with at least the following keywords:

================  ==========================================================================================================================================================================================================
keyword           description
================  ==========================================================================================================================================================================================================
OBSERVATORY_NAME  name of the observatory where the observations were taken. This is important for instance for coordinate transformations between celestial (i.e. RA/dec) and terrestrial (i.e. az/alt) coordinate systems.
MJDREFI           reference time for other times in the list (i.e. TIME_START/TIME_STOP). Integer value in mean julian days.
MJDREFF           fraction of integer value defined in MJDREFI.
================  ==========================================================================================================================================================================================================

Extra user defined header entries are allowed; Gammapy will ignore them.

TODO: should the observation list have already a header? Or should the header values be defined as columns, then interpreted correspondingly when stored into an ObservationTable? (I might be mixing up 2 things: obs lists and obs tables here? or should they have the same format?)

TODO: should the observation list already define the times in this ``MJDREFI + MJDREFF`` format? Or should it be converted once its read into an ObservationTable?

TODO: ``TEL_LIST``: incompatibility wit CSV format, since it's a CSV list of tels???!!!

TODO: how to biuld sphinx only for a specific module or file?

TODO: add an example table?!!! (and header? how?)

TODO: restructure (as requested in PR)!!!
