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

Usually it has many more columns with information about each observation. A list of
Gammapy supported columns is:

================  ===========================================================================  =========
column name       description                                                                  required?
================  ===========================================================================  =========
OBS_ID            observation ID as an integer                                                 yes
RA                pointing position right ascension in equatorial (ICRS) coordinates           yes?
DEC               pointing position declination in equatorial (ICRS) coordinates               yes?
AZ                average azimuth angle during the observation                                 no
ALT               average altitude angle during the observation                                no
MUON_EFFICIENCY   average muon efficiency of the telescopes                                    yes?
TIME_START        start time of the observation stored as number of seconds after [MET]_ or    yes?
                  as absolute times in UTC (see description of header keyword `TIME FORMAT`)
TIME_STOP         end time of the observation in the same format as TIME_START                 no
TIME_OBSERVATION  duration of the observation                                                  no
TIME_LIVE         duration of the observation without dead time                                no
TRIGGER_RATE      average trigger rate of the system                                           no
MEAN_TEMPERATURE  average temperature of the atmosphere                                        no
N_TELS            number of telescopes participating in the observation                        yes?
TEL_LIST          string with a [CSV]_ list of telescope IDs participating in the observation  yes?
QUALITY           data quality; recommended: "spectral" or "detection"                         no
================  ===========================================================================  =========

Extra user-defined columns are allowed; Gammapy will ignore them.

In order for the extra columns to have full meaning the following is needed:

 * Extra row right after the column name, specifying the unit of the quantity listed on each column.
 * A header with at least the following keywords:

================  ===========================================================================  =========
keyword           description                                                                  required?
================  ===========================================================================  =========
OBSERVATORY_NAME  name of the observatory where the observations were taken. This is           no
                  important for instance for coordinate transformations between celestial
                  (i.e. RA/dec) and terrestrial (i.e. az/alt) coordinate systems
MJDREFI           reference time: integer value in mean julian days; details in                yes?
                  :ref:`time_handling`
MJDREFF           reference time: fraction of integer value defined in MJDREFI; details in     yes?
                  :ref:`time_handling`
TIME_FORMAT       format in which times are stored: `absolute` (UTC) or `relative` ([MET]_);   yes?
                  see details for both formats in :ref:`time_handling`
================  ===========================================================================  =========

Extra user-defined header entries are allowed; Gammapy will ignore them.


Example
-------
The tool `~gammapy.datasets.make_test_observation_table` can generate a `~gammapy.obs.ObservationTable`
with dummy values.

Header (metadata)::

   {u'MJDREFI': 55197.0, u'OBSERVATORY_NAME': u'HESS', u'MJDREFF': 0.0}

Table:

+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|OBS_ID|TIME_OBSERVATION|TIME_LIVE|  TIME_START |  TIME_STOP  |      AZ     |     ALT     |      RA     |     DEC      |N_TELS|MUON_EFFICIENCY|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|      |       s        |    s    |      s      |      s      |     deg     |     deg     |     deg     |     deg      |      |               |
+======+================+=========+=============+=============+=============+=============+=============+==============+======+===============+
|     1|          1800.0|   1500.0|118890428.352|118892228.352|130.926874751|49.6209457026|96.3849089136|-43.6914197077|     3| 0.814535992712|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|     2|          1800.0|   1500.0|112242990.559|112244790.559|272.213179564|70.2673929472| 339.00128923|-21.1698098192|     3| 0.976469816749|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|     3|          1800.0|   1500.0| 66444741.854| 66446541.854|346.848818939| 46.138583188|162.086175054| 19.6398873974|     4| 0.920096961383|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|     4|          1800.0|   1500.0|22388716.9244|22390516.9244|300.850748052|55.1330124055|32.9474858892|-3.19910057294|     3| 0.678431411337|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|     5|          1800.0|   1500.0|135469063.888|135470863.888|355.160931662| 48.734744852|197.123663537| 17.9411145072|     4|  0.77879533822|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|     6|          1800.0|   1500.0|21857681.6916|21859481.6916|124.846967209| 78.859585347| 14.162859563|-29.3419432185|     4| 0.709642622408|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|     7|          1800.0|   1500.0| 57554741.106| 57556541.106|268.541714486|48.8489560299|64.8265458802|-18.2634404823|     3| 0.908426763354|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|     8|          1800.0|   1500.0|19181027.9045|19182827.9045|120.558129392| 49.663761361| 24.791511978|-37.1789681608|     4| 0.980162662473|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|     9|          1800.0|   1500.0|120447694.583|120449494.583| 132.10271454|78.7455993174|89.7950895353|-30.5128854184|     3| 0.807695978946|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+
|    10|          1800.0|   1500.0|144207430.361|144209230.361|323.562657045|45.4005803262|324.596045439| 13.6761217326|     3| 0.694201696626|
+------+----------------+---------+-------------+-------------+-------------+-------------+-------------+--------------+------+---------------+

A test observation list file in fits format is located in the
`~gammapy-extra` repository
(`test_observation_table.fits <https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/obs/test_observation_table.fits>`_).
