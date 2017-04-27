.. _time:

*******************************************
Time handling and analysis (`gammapy.time`)
*******************************************

.. currentmodule:: gammapy.time

Introduction
============

`gammapy.time` contains methods for timing analysis.

TODO: explain `gammapy.utils.time` and why it's separate.

At the moment there isn't a lot of functionality yet ... contributions welcome!


Getting Started
===============

.. _time-lc:

Lightcurve
----------

The `~gammapy.time.LightCurve` class can be used to read a lightcurve and plot it:

.. TODO: make better example from file or Fermi-LAT
.. >>> lc = LightCurve.read('$GAMMAPY_EXTRA/todo/make/example-lightcurve.fits.gz')

.. plot::
   :include-source:

    >>> from gammapy.time import LightCurve
    >>> lc = LightCurve.simulate_example()
    >>> lc.plot()

Here's how to compute some summary statistics for the lightcurve:

.. code-block:: python

    >>> lc['FLUX'].mean()
    <Quantity 5.25 1 / (cm2 s)>

TODO: please help extend the functionality and examples for `~gammapy.time.LightCurve`!

.. _time-variability:

Variability test
----------------

The `~gammapy.time.exptest` function can be used to compute the significance
of variability (compared to the null hypothesis of constant rate)
for a list of event time differences.

Here's an example how to use the `~gammapy.time.random_times` helper
function to simulate a `~astropy.time.TimeDelta` array for a given constant rate
and use `~gammapy.time.exptest` to assess the level of variability (0.11 sigma in this case,
not variable):

.. code-block:: python

    >>> from astropy.units import Quantity
    >>> from gammapy.time import random_times, exptest
    >>> rate = Quantity(10, 'Hz')
    >>> time_delta = random_times(size=100, rate=rate, return_diff=True, random_state=0)
    >>> mr = exptest(time_delta)
    >>> print(mr)
    0.11395763079


See ``examples/example_exptest.py`` for a longer example.

TODO: apply this to the 2FHL events and check which sources are variable as a nice example.

.. code-block:: python

    from gammapy.data import EventList
    from gammapy.time import exptest
    events = EventList.read('$GAMMAPY_EXTRA/datasets/fermi_2fhl/2fhl_events.fits.gz ', hdu='EVENTS')
    # TODO: cone select events for 2FHL catalog sources, compute mr for each and print 10 most variable sources


Other codes
===========

Where possible we shouldn't duplicate timing analysis functionality
that's already available elsewhere. In some cases it's enough to refer
gamma-ray astronomers to other packages, sometimes a convenience wrapper for
our data formats or units might make sense.

Some references (please add what you find useful

* https://github.com/astroML/gatspy
* https://github.com/matteobachetti/MaLTPyNT
* https://github.com/nanograv/PINT
* http://www.astroml.org/modules/classes.html#module-astroML.time_series
* http://www.astroml.org/book_figures/chapter10/index.html
* http://docs.astropy.org/en/latest/api/astropy.stats.bayesian_blocks.html
* https://github.com/samconnolly/DELightcurveSimulation
* https://github.com/cokelaer/spectrum
* https://github.com/YSOVAR/YSOVAR
* https://nbviewer.ipython.org/github/YSOVAR/Analysis/blob/master/TimeScalesinYSOVAR.ipynb


Reference/API
=============

.. automodapi:: gammapy.time
    :no-inheritance-diagram:
