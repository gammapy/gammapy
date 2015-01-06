.. _detect:

*****************************************
Source detection tools (`gammapy.detect`)
*****************************************

.. currentmodule:: gammapy.detect

Introduction
============

`gammapy.detect` holds source detection methods that turn event lists or images or cubes
into source catalogs. 

* TODO: Describe references: [Stewart2009]_
* TODO: general intro
* TODO: describe the ``gammapy-iterative-source-detect`` tool.
* TODO: describe the ``gammapy-image-decompose-a-trous`` tool.


Getting Started
===============

Computation of TS maps
----------------------

The `gammapy.detect` module includes a high performance `~gammapy.detect.compute_ts_map` function to
compute test statistics (TS) maps for Gamma-Ray survey data. The implementation is based on the method 
described in [Stewart2009]_.

Assuming a certain source morphology, which can be defined by any `astropy.convolution.Kernel2D`
instance, the amplitude of the morphology model is fitted at every pixel of the input data using a 
Poisson maximum likelihood procedure. As input data a counts, background and exposure map have to be provided.
Based on the best fit flux amplitude, the change in TS, compared to the null hypothesis is computed
using `~gammapy.stats.cash` statistics.

To optimize the performance of the code, the fitting procedure is simplified by finding roots
of the derivative of the fit statistics with respect to the flux amplitude. This approach is
described in detail in Appendix A of [Stewart2009]_. To further improve the performance,
Pythons's `multiprocessing` facility is used.

In the following the computation of a TS map for prepared Fermi survey data, which is provided in 
`gammapy-extra <https://github.com/gammapy/gammapy-extra/tree/master/datasets/fermi_survey>`_, shall be demonstrated:

.. code-block:: python

	from astropy.io import fits
	from astropy.convolution import Gaussian2DKernel
	from gammapy.detect import compute_ts_map
	hdu_list = fits.open('all.fits.gz')
	kernel = Gaussian2DKernel(5)
	result = compute_ts_map(hdu_list['On'].data, hdu_list['Background'].data,
							hdu_list['ExpGammaMap'].data, kernel, threshold=0)

The option ``threshold=0`` sets a minimal required TS value based on the initial flux estimate, that the pixel is
processed at all. The function returns a `~gammapy.detect.TSMapResult` object, that bundles all relevant
data. E.g. the time needed for the TS map computation can be checked by:

.. code-block:: python

	print(result.runtime)

The TS map itself can bes accessed using the ``ts`` attribute of the `~gammapy.detect.TSMapResult` object:

.. code-block:: python

	print(result.ts.max())

Command line tool
-----------------

Gammapy also provides a command line tool ``gammapy-ts-image`` for TS map computation, which can be run
on the Fermi example dataset by:

.. code-block:: bash

	$ cd gammapy-extra/datasets/fermi_survey
	$ gammapy-ts-image all.fits.gz --threshold 0 --scale 0 

The command line tool additionally requires a psf json file, where the psf shape is defined by the parameters
of a triple Gaussian model. See also `gammapy.irf.multi_gauss_psf_kernel`. By default the command line tool uses
a Gaussian source kernel, where the width in degree can be defined by the ``--scale`` parameter. When setting
``--scale 0`` only the psf is used as source model, which is the preferred setting to detect point sources.
When using scales that are larger than five times the binning of the data, the data is sampled down and later
sampled up again to speed up the performance. See `~gammapy.image.downsample_2N` and `~gammapy.image.upsample_2N` for details.   

Furthermore it is possible to compute residual TS maps. Using the following options:
 
.. code-block:: bash

	$ gammapy-ts-image all.fits.gz --threshold 0 --scale 0 --residual --model model.fits.gz 

When ``--residual`` is set an excess model must be provided using the ``--model`` option.

Reference/API
=============

.. automodapi:: gammapy.detect
    :no-inheritance-diagram:
