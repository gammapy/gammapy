.. _cube:

************************************
Cube Style Analysis (`gammapy.cube`)
************************************

.. currentmodule:: gammapy.cube

Introduction
============

The `~gammapy.cube` module bundles functionality for combined spatial and
spectral analysis (cube style analysis) of gamma-ray sources.

Some information on cube style analysis in gamma-ray astronomy can be found here:

* `Cube style analysis for Cherenkov telescope data`_
* `Classical analysis in VHE gamma-ray astronomy`_

.. _Cube style analysis for Cherenkov telescope data: https://github.com/gammapy/PyGamma15/blob/gh-pages/talks/analysis-cube/2015-11-16_PyGamma15_Eger_Cube_Analysis.pdf
.. _Classical analysis in VHE gamma-ray astronomy: https://github.com/gammapy/PyGamma15/blob/gh-pages/talks/analysis-classical/2015-11-16_PyGamma15_Terrier_Classical_Analysis.pdf


Getting Started
===============

Use `~gammapy.cube.SkyCube` to read a Fermi-LAT diffuse model cube::

    >>> from gammapy.cube import SkyCube
    >>> filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/fermi/gll_iem_v02_cutout.fits'
    >>> cube = SkyCube.read(filename, format='fermi-background')
    >>> print(cube)
    Sky cube flux with shape=(30, 21, 61) and unit=1 / (cm2 MeV s sr):
     n_lon:       61  type_lon:    GLON-CAR         unit_lon:    deg
     n_lat:       21  type_lat:    GLAT-CAR         unit_lat:    deg
     n_energy:    30  unit_energy: MeV

Use the cube methods to do computations::

    import astropy.units as u
    emin, emax = [1, 10] * u.GeV
    image = cube.sky_image_integral(emin=emin, emax=emax)
    image.show('ds9')

TODO: also show how to work with counts and exposure cube using the example at ``test_datasets/unbundled/fermi``
(or make a better one).

Reference/API
=============

.. automodapi:: gammapy.cube
    :no-inheritance-diagram: