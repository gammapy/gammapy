.. include:: ../references.txt

.. _detect:

*************************
detect - Source detection
*************************

.. currentmodule:: gammapy.detect

Introduction
============

The `gammapy.detect` submodule includes low level functions to compute
significance and test statistics images as well as some high level source
detection method prototypes.

Detailed description of the methods can be found in [Stewart2009]_ and
[LiMa1983]_.

Note that in Gammapy maps are stored as Numpy arrays, which implies that it's
very easy to use `scikit-image`_ or `photutils`_ or other packages that have
advanced image analysis and source detection methods readily available.

Computation of TS images
========================

.. gp-extra-image:: detect/fermi_ts_image.png
    :width: 100%

Test statistics image computed using `~gammapy.detect.TSMapEstimator` for an
example Fermi dataset.

The `gammapy.detect` module includes a high performance
`~gammapy.detect.TSMapEstimator` class to compute test statistics (TS) images
for gamma-ray survey data. The implementation is based on the method described
in [Stewart2009]_.

Assuming a certain source morphology, which can be defined by any
`astropy.convolution.Kernel2D` instance, the amplitude of the morphology model
is fitted at every pixel of the input data using a Poisson maximum likelihood
procedure. As input data a counts, background and exposure images have to be
provided. Based on the best fit flux amplitude, the change in TS, compared to
the null hypothesis is computed using `~gammapy.stats.cash` statistics.

To optimize the performance of the code, the fitting procedure is simplified by
finding roots of the derivative of the fit statistics with respect to the flux
amplitude. This approach is described in detail in Appendix A of [Stewart2009]_.
To further improve the performance, Pythons's `multiprocessing` facility is
used.

The following example shows how to compute a TS image for Fermi-LAT survey data:

.. code-block:: python

    from astropy.convolution import Gaussian2DKernel
    from gammapy.detect import TSMapEstimator
    from gammapy.maps import Map

    filename = '$GAMMAPY_DATA/fermi_survey/all.fits.gz'
    maps = {}
    maps['counts'] = Map.read(filename, hdu='counts')
    maps['exposure'] = Map.read(filename, hdu='exposure')
    maps['background'] = Map.read(filename, hdu='background')

    kernel = Gaussian2DKernel(5)
    ts_estimator = TSMapEstimator()
    result = ts_estimator.run(maps, kernel)

The function returns an dictionary, that bundles all resulting maps. E.g. here's
how to find the largest TS value:

.. code-block:: python

    import numpy as np
    np.nanmax(result['ts'].data)

Computation of Li & Ma significance images
==========================================

The method derived by [LiMa1983]_ is one of the standard methods to determine
detection significances for gamma-ray sources. Using the same prepared Fermi
dataset as above, the corresponding images can be computed using the
`~gammapy.detect.compute_lima_image` function:

.. code-block:: python

    from astropy.convolution import Tophat2DKernel
    from gammapy.maps import Map
    from gammapy.detect import compute_lima_image
    filename = '$GAMMAPY_DATA/fermi_survey/all.fits.gz'
    counts = Map.read(filename, hdu='COUNTS')
    background = Map.read(filename, hdu='BACKGROUND')
    kernel = Tophat2DKernel(5)
    result = compute_lima_image(counts, background, kernel)

The function returns a dictionary, that bundles all resulting images such as
significance, flux and correlated counts and excess images.

Using `gammapy.detect`
======================

:ref:`tutorials` that show examples using ``gammapy.detect``:

* :gp-extra-notebook:`detect_ts`

Reference/API
=============

.. automodapi:: gammapy.detect
    :no-inheritance-diagram:
    :include-all-objects:
