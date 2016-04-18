.. _tutorials-npred:

Fermi-LAT diffuse model predicted counts image
==============================================

The `~gammapy.cube.SkyCube` class allows for image-based analysis in energy
bands. In particular, similar functionality to gtmodel in the Fermi Science
tools [FSSC2013]_ is offered in `~gammapy.data.compute_npred_cube`
which generates a predicted instrument PSF-convolved counts cube based on a
provided background model. Unlike the science tools, this implementation is
appropriate for use with large regions of the sky. 


Predicting Counts
-----------------

The example script below computes the Fermi PSF-convolved predicted counts map
using `~gammapy.cube.SkyCube`. This is then used to produce a Li & Ma significance
image [LiMa1983]_. The left image shows the significance image,
while a comparison against the significance image
produced using the Fermi Science tools is shown on the right. These results are
for the Vela region for energies between 10 and 500 GeV.

.. literalinclude:: npred_general.py

.. plot:: tutorials/npred/npred_convolved_significance.py
   :include-source:
   
   
Checks
------

For small regions, the predicted counts cube and significance images may be
checked against the gtmodel output. The Vela region shown above is taken in
this example in one energy band with the parameters outlined in the
`README file for FermiVelaRegion
<https://github.com/gammapy/gammapy-extra/blob/master/datasets/vela_region/README.rst>`_.

Images for the predicted background counts in this region in the Gammapy case
(left) and Fermi Science Tools gtmodel case (center) are shown below, based on
the differential flux contribution of the Fermi diffuse model gll_iem_v05_rev1.fit.
The image on the right shows the ratio. **Note that the colorbar scale applies only to
the ratio image.**

.. plot:: tutorials/npred/npred_convolved.py
   :include-source:

We may compare these against the true counts observed by Fermi LAT in this region
for the same parameters:

 * True total counts: 1551
 * Fermi Tools gtmodel predicted background counts: 265
 * Gammapy predicted background counts: 282
