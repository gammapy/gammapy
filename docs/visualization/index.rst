.. include:: ../references.txt

.. _visualization:

******************************************
visualization - Plotting and visualization
******************************************

.. currentmodule:: gammapy.visualization

Getting started
===============

`gammapy.visualization` provides a few helper functions and classes to create
publication-quality images.

Colormaps
---------

The following example shows how to plot images using colormaps that are commonly
used in gamma-ray astronomy (`colormap_hess` and `colormap_milagro`).

.. plot:: visualization/colormap_example.py

Survey panel plots
------------------

The `MapPanelPlotter` allows to split e.g. a galactic plane survey image with
a large aspect ratio into multiple panels. Here is a short example:

.. plot::

    import matplotlib.pyplot as plt
    from astropy.coordinates import SkyCoord, Angle
    from gammapy.maps import Map
    from gammapy.data import EventList
    from gammapy.visualization import MapPanelPlotter

    skydir = SkyCoord("0d", "0d", frame="galactic")
    counts = Map.create(skydir=skydir, width=(180, 10), frame="galactic")

    events = EventList.read("$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_events_selected.fits.gz")
    counts.fill_events(events)

    smoothed = counts.smooth("0.1 deg")

    fig = plt.figure(figsize=(12, 6))
    xlim, ylim = Angle(["90d", "270d"]), Angle(["-5d", "5d"])
    plotter = MapPanelPlotter(figure=fig, xlim=xlim, ylim=ylim, npanels=3)

    axes = plotter.plot(smoothed, vmax=10)
    plt.show()



Reference/API
=============

.. automodapi:: gammapy.visualization
    :no-inheritance-diagram:
    :include-all-objects:
