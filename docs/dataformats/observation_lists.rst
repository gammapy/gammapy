.. _dataformats_observation_lists:

Observation lists
=================

Run lists specify a list of observation runs to be processed by a given tool.

In the end the tool will usually generate input and output data filenames from
the information in the run list. These filenames can either be given directly
or there can be naming conventions, e.g. the event list for run ``42`` could be stored
at ``$TEV_DATA/RUN_000042/Events.fits``.


CVS format
----------

We use the `CSV <http://en.wikipedia.org/wiki/Comma-separated_values>`_ (comma-separated values) format for run lists.
This has the advantage that everyone can work with whatever tool they like. Here's some good options:

* `LibreOffice Calc <http://www.libreoffice.org/features/calc/>`_ 
* `TOPCAT <http://www.star.bris.ac.uk/~mbt/topcat/>`_ or `STILTS <http://www.star.bris.ac.uk/~mbt/stilts/>`_
* `csvkit <https://csvkit.readthedocs.org/en/latest/>`_
* Your own script using e.g the `Python standard library CSV module <http://docs.python.org/2/library/csv.html>`_ or `pandas <http://pandas.pydata.org>`_

A run list must have at least a column called ``Run``::
 
   Run
   42
   43

Usually it has many more columns with information about each run::
 
   Run,Telescope_Pattern,Date,Duration,GLON,GLAT,Target,Zenith
   1234,24,2013-03-22 14:32,1832,83.7,-5.2,Crab Nebula,32

Special column names that the Gammapy analysis tools understand:

* ``Run`` --- Run number (int)
* ``Telescope_Pattern`` --- Binary pattern describing which telescopes participated in the run
* ``Date`` --- Date of observation
* ``RA``, ``DEC`` or ``GLON``, ``GLAT`` -- Pointing position in Equatorial (ICRS) or Galactic coordinates

XML format
----------

GammaLib / ctools uses an "observation definition" XML format described
`here <http://gammalib.sourceforge.net/user_manual/modules/obs.html#describing-observations-using-xml>`__.
