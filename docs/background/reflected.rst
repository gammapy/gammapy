.. include:: ../references.txt

.. _region_reflected:

****************************
Reflected regions background
****************************

.. currentmodule:: gammapy.background

Overview
--------

This technique is used in classical Cherenkov astronomy for the 1D spectral
extraction. A region on the sky, the ON region, is chosen to select events
around the studied source position. In the absence of a solid template of the
residual hadronic background, a classical method to estimate it is the so-called
Reflected Region Background. The underlying assumption is that the background is
approximately purely radial in the field-of-view. A set of OFF counts is found
in the observation, by rotating the ON region selected around the pointing
position. To avoid that the reflectd regions contain actual gamma-ray signal
from other objects, one has to remove the gamma-ray bright parts of the
field-of-view with a exclusion mask. Details on the reflected regions method can
be found in [Berge2007]_

The extraction of the ON and OFF events from the `~gammapy.data.EventList` of a
set of observations is performed by the
`~gammapy.background.ReflectedRegionsBackgroundEstimator`. The latter uses the
`~gammapy.background.ReflectedRegionsFinder` to create reflected regions for a
given circular on region and exclusion mask.

Using regions
-------------

The on region is a `~regions.SkyRegion`. It is typically a circle
(`~regions.CircleSkyRegion`) for pointlike source analysis, but can be a more
complex region such as a `~regions.CircleAnnulusSkyRegion` a
`~regions.EllipseSkyRegion`, a `~regions.RectangleSkyRegion` etc.

The following example shows how to create such regions:

.. plot:: background/create_region.py
    :include-source:

The reflected region finder
---------------------------

The following example illustrates how to create reflected regions for a given
circular on region and exclusion mask using the
`~gammapy.background.ReflectedRegionsFinder`. In particular, it shows how to
change the minimal distance between the ON region and the reflected regions.
This is useful to limit contamination by events leaking out the ON region. It
also shows how to change the minimum distance between adjacent regions as well
as the maximum number of reflected regions.

.. plot:: background/make_reflected_regions.py
    :include-source:

Using the reflected background estimator
----------------------------------------

In practice, the user does not usually need to directly interact with the
`~gammapy.background.ReflectedRegionsFinder`. This actually is done via the
`~gammapy.background.ReflectedRegionsBackgroundEstimator`, which extracts the ON
and OFF events for an `~gammapy.data.Observations` object. The last example
shows how to run it on a few observations with a rectangular region.

.. plot:: background/make_rectangular_reflected_background.py
    :include-source:

The following notebook shows an example using
`~gammapy.background.ReflectedRegionsBackgroundEstimator` to perform a spectral
extraction and fitting:

* :gp-notebook:`spectrum_analysis`


