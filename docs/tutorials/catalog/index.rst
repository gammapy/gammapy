.. _tutorials-catalog:

Catalog & Simulation Images
===========================

The `~gammapy.image.catalog_image` method allows the production of
single energy-band 2D images from point source catalogs, either true catalogs
(e.g. 1FHL or 2FGL) or source catalogs of simulated galaxies (produced with
`~gammapy.astro.population`). Examples of these two use-cases are included below.

Source Catalog Images
---------------------

The example script below produces a point source catalog image from the published 
1FHL Fermi Source Catalog from 10 to 500 GeV. Fluxes are filled into each pixel
corresponding to source Galactic Latitude and Longitude, and then convolved with
the Fermi PSF in this energy band.

.. _fermi-1fhl-image:

.. plot:: tutorials/catalog/source_image_demo.py
	:include-source:
   
Simulated Catalog Images
------------------------

In this case, a galaxy is simulated with `~gammapy.astro.population` to produce a
source catalog. This is then converted into an image.

.. plot:: tutorials/catalog/simulated_image_demo.py
	:include-source:
	
Caveats & Future Developments
-----------------------------

It should be noted that the current implementation does not support:

* The inclusion of extended sources
* Production of images in more than one energy band
