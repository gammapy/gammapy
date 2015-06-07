.. _time:

********************************
Timing analysis (`gammapy.time`)
********************************

.. currentmodule:: gammapy.time

Introduction
============

`gammapy.time` contains methods for timing analysis.

At the moment there's almost no functionality here.
Any contributions welcome, e.g.

* utility functions for time computations
* lightcurve analysis functions
* pulsar periodogram
* event dead time calculations and checks for constant rate

Although where possible we shouldn't duplicate timing analysis functionality
that's already available elsewhere. In some cases it's enough to refer
gamma-ray astronomers to other packages, sometimes a convenience wrapper for
our data formats or units might make sense.

Some references (please add what you find useful

* https://github.com/matteobachetti/MaLTPyNT
* https://github.com/nanograv/PINT
* http://www.astroml.org/modules/classes.html#module-astroML.time_series
* http://www.astroml.org/book_figures/chapter10/index.html
* http://astropy.readthedocs.org/en/latest/api/astropy.stats.bayesian_blocks.html
* https://github.com/samconnolly/DELightcurveSimulation
* https://github.com/cokelaer/spectrum
* https://github.com/YSOVAR/YSOVAR
* http://nbviewer.ipython.org/github/YSOVAR/Analysis/blob/master/TimeScalesinYSOVAR.ipynb

Getting Started
===============

TODO: document

Reference/API
=============

.. automodapi:: gammapy.time
    :no-inheritance-diagram:
