.. include:: ../references.txt

.. _region_reflected:

****************************
Reflected regions background
****************************

.. currentmodule:: gammapy.background

This technique is used in classical Cherenkov astronomy for the 1D spectral extraction.
A region on the sky, the ON region, is chosen to select events around the studied source
position. In the absence of a solid template of the residual hadronic background, a classical
method to estimate it is the so-called Reflected Region Background. The underlying assumption
is that the background is approximately purely radial in the field-of-view. A set of OFF counts
is found in the observation, by rotating the ON region selected around the pointing position.
To avoid that the reflectd regions contain actual gamma-ray signal from other objects, one has
to remove the gamma-ray bright parts of the field-of-view with a exclusion mask.
Details on the reflected regions method can be found in [Berge2007]_

The extraction of the ON and OFF events from the `~gammapy.data.EventList` of a set of
observations is performed by the `~gammapy.background.ReflectedBackgroundEstimator`.
The latter uses the `~gammapy.background.ReflectedRegionsFinder` to create reflected
regions for a given circular on region and exclusion mask.

The ON region is a `~regions.SkyRegion`. It is typically a circle (`~regions.CircleSkyRegion`)
for pontlike source analysis, but can be a more complex region such as a `~regions.CircleAnnulusSkyRegion`
a `~regions.EllipseSkyRegion`, a `~regions.RectangleSkyRegion` etc.

The following example shows how to create such regions:
.. plot:: background/create_regions.py
    :include-source:

The following example illustrates how to create reflected regions for a given
circular on region and exclusion mask using the `~gammapy.background.ReflectedRegionsFinder`.

.. plot:: background/make_reflected_regions.py
    :include-source:
