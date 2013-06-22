**************************************
Observation bookkeeping  (`tevpy.obs`)
**************************************

.. currentmodule:: tevpy.obs

Introduction
============

`tevpy.obs` contains methods to do the bookkeeping for processing multiple observations.

In TeV astronomy an observation (a.k.a. a run) means pointing the telescopes at some
position on the sky (fixed in celestial coordinates, not in horizon coordinates)
for a given amount of time (e.g. half an hour) and switching the central trigger on.

The total dataset for a given target will usually consist of a few to a few 100 runs
and some book-keeping is required when running the analysis.


Getting Started
===============

Run list format
---------------

We use the `CSV <http://en.wikipedia.org/wiki/Comma-separated_values>`_ (comma-separated values) format for run lists.
This has the advantage that everyone can work with whatever tool they like. Here's some good options:

* `LibreOffice Calc <http://www.libreoffice.org/features/calc/>`_ 
* `TOPCAT <http://www.star.bris.ac.uk/~mbt/topcat/>`_ or `STILTS <http://www.star.bris.ac.uk/~mbt/stilts/>`_
* `csvkit <https://csvkit.readthedocs.org/en/latest/>`_
* Your own script using e.g the `Python standard library CSV module <http://docs.python.org/2/library/csv.html>`_ or `pandas <http://pandas.pydata.org>`_

A run list must have at least a column called `Run`:
 
   Run
   42
   43

Usually it has many more columns with information about each run:
 
   Run,Telescope_Pattern,Date,Duration,GLON,GLAT,Target,Zenith
   1234,24,2013-03-22 14:32,1832,83.7,-5.2,Crab Nebula,32

Special column names that the `tevpy` analysis tools understand:

* `Run` --- Run number (int)
* `Telescope_Pattern` --- Binary pattern describing which telescopes participated in the run
* `Date` --- Date of observation
* `RA`, `DEC` or `GLON`, `GLAT` -- Pointing position in Equatorial (ICRS) or Galactic coordinates


findruns
--------

The `findruns` command line tool can be used to select a subset of runs from a given run list.

Examples:

   Find all runs within 5 deg of the Galactic center
   $ findruns cone --x 0 --y 0 --r 5 --system galactic --in all_runs.lis --out galactic_center_runs.lis
   
   Select all runs in a box along the Galactic plane (GLON = -20 .. +20 deg, GLAT = -3 .. +3 deg)
   $ findruns box --x 0 --y 0 --dx 20 --dy 3 --system galactic --in all_runs.lis  --out galactic_plane_runs.lis

   Select all runs in a given date range (can of course be combined with other selections shown above)
   $ findruns --date_min 2010-04-26 --date_max "2010-04-29 12:42"  --in all_runs.lis --out run_042_to_100.lis

   Select all runs in a given run range (can of course be combined with other selections shown above)
   $ findruns --run_min 42 --run_max 100 --in all_runs.lis --out run_042_to_100.lis
   
Using `findruns` is easy, you don't have to remember all the options, just type

   $ findruns --help

at the command line or read the usage help `here <TODO>`_

..note:: Noel Dawe's `goodruns` tool for `ATLAS <http://atlas.ch>`_ run selection
      (`docs <http://ndawe.github.io/goodruns/>`, `code <https://github.com/ndawe/goodruns>`_)
      is a nice example for a run selection tool.


.. automodapi:: tevpy.obs
    :no-inheritance-diagram:
