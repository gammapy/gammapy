# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions for reading / writing model parameters to JSON files.

At the moment you can have any number of Gaussians.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.extern import six
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma
from ...utils.random import get_random_state

__all__ = [
    'make_test_model',
    'read_json',
    'MorphModelImageCreator',
]

__doctest_skip__ = ['MorphModelImageCreator']


class MorphModelImageCreator(object):
    """Create model images from a HGPS pipeline source config file.

    Uses astropy to evaluate the source model, with oversampling or integrating
    over pixels.

    Parameters
    ----------
    cfg_file : str
        Config file with all the sources listed.
    exposure : str
        Fits image file with the exposure.
    psf_file : str (optional)
        Json file with PSF information.
    background : str (optional)
        Fits image file with the background.
    apply_psf : bool
        Whether the psf should be applied.
    compute_excess : bool
        Whether to compute an excess image.
    flux_factor : float
        Flux conversion factor.

    Examples
    --------
    Here is an example how to use `MorphModelImageCreator`:

    >>> from gammapy.image.models import MorphModelImageCreator
    >>> model_image_creator = MorphModelImageCreator(cfg_file='input_sherpa.cfg',
    ...                                              exposure='exposure.fits',
    ...                                              psf_file='psf.json')
    >>> model_image_creator.evaluate_model(mode='center')
    >>> model_image_creator.save('model_image.fits')
    """

    def __init__(self, cfg_file, exposure, psf_file=None, apply_psf=True,
                 background=None, flux_factor=1e-12, compute_excess=True):
        self.cfg_file = cfg_file
        self.exposure = fits.getdata(exposure)
        self.header = fits.getheader(exposure)
        self._apply_psf = apply_psf
        self._flux_factor = flux_factor
        self._compute_excess = compute_excess
        if psf_file is not None:
            self.psf_file = psf_file
        if background is not None:
            if isinstance(background, six.string_types):
                self.background = fits.getdata(background)
            elif isinstance(background, (int, float)):
                self.background = np.ones_like(self.exposure)

    def _setup_model(self):
        """Setup a list of source models from an ``input_sherpa.cfg`` config file.
        """
        self.source_models = []

        # Read config file
        from astropy.extern.configobj.configobj import ConfigObj
        cfg = ConfigObj(self.cfg_file, file_error=True)

        # Set up model
        from astropy.modeling.models import Gaussian2D

        for source in cfg.keys():
            # TODO: Add other source models
            if cfg[source]['Type'] != 'NormGaussian':
                raise ValueError('So far only normgauss2d models can be handled.')
            sigma = gaussian_fwhm_to_sigma * float(cfg[source]['fwhm'])
            ampl = float(cfg[source]['ampl']) * 1 / (2 * np.pi * sigma ** 2)
            xpos = float(cfg[source]['xpos']) - 1
            ypos = float(cfg[source]['ypos']) - 1
            source_model = Gaussian2D(ampl, xpos, ypos, x_stddev=sigma, y_stddev=sigma)
            self.source_models.append(source_model)

    def evaluate_model(self, **kwargs):
        """Evaluate model by oversampling or taking the value at the center of the pixel.
        """
        self._setup_model()
        self.model_image = np.zeros_like(self.exposure, dtype=np.float64)

        from astropy.convolution import utils
        height, width = self.exposure.shape

        for source_model in self.source_models:
            source_model_image = utils.discretize_model(source_model,
                                                        (0, width), (0, height), **kwargs)
            self.model_image += source_model_image

        if self._compute_excess:
            self.model_image = self.model_image * self.exposure

        if self._apply_psf:
            psf = self._create_psf(**kwargs)
            from astropy.convolution import convolve
            self.model_image = convolve(self.model_image, psf)
        self.model_image *= self._flux_factor

    def _create_psf(self, **kwargs):
        """Set up psf model using `astropy.convolution`.
        """
        # Read psf info
        import json
        psf_data = json.load(open(self.psf_file))

        # Convert sigma and amplitude
        sigma_1 = gaussian_fwhm_to_sigma * psf_data['psf1']['fwhm']
        sigma_2 = gaussian_fwhm_to_sigma * psf_data['psf2']['fwhm']
        sigma_3 = gaussian_fwhm_to_sigma * psf_data['psf3']['fwhm']
        ampl_1 = psf_data['psf1']['ampl'] * 2 * np.pi * sigma_1 ** 2
        ampl_2 = psf_data['psf2']['ampl'] * 2 * np.pi * sigma_2 ** 2
        ampl_3 = psf_data['psf3']['ampl'] * 2 * np.pi * sigma_3 ** 2

        # Setup kernels
        from astropy.convolution import Gaussian2DKernel
        gauss_1 = Gaussian2DKernel(sigma_1, **kwargs)
        gauss_2 = Gaussian2DKernel(sigma_2, **kwargs)
        gauss_3 = Gaussian2DKernel(sigma_3, **kwargs)
        psf = gauss_1 * ampl_1 + gauss_2 * ampl_2 + gauss_3 * ampl_3
        psf.normalize()
        return psf

    def save(self, filename, **kwargs):
        """Save model image to file."""
        hdu_list = []
        prim_hdu = fits.PrimaryHDU(self.model_image, header=self.header)
        hdu_list.append(prim_hdu)
        fits_hdu_list = fits.HDUList(hdu_list)
        fits_hdu_list.writeto(filename, **kwargs)

        if hasattr(self, 'measurements'):
            hdu_list = []
            prim_hdu = fits.PrimaryHDU(self.measurements[0], header=self.header)
            hdu_list.append(prim_hdu)

            for image in self.measurements[1:]:
                hdu = fits.ImageHDU(image)
                hdu_list.append(hdu)
            fits_hdu_list = fits.HDUList(hdu_list)
            fits_hdu_list.writeto('counts_' + filename, **kwargs)

    def fake_counts(self, N, random_state='random-seed', **kwargs):
        """Fake measurement data by adding Poisson noise to the model image.

        Parameters
        ----------
        N : int
            Number of measurements to fake.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        if not self._compute_excess:
            self.model_image = self.model_image * self.exposure
        if not self._apply_psf:
            psf = self._create_psf(**kwargs)
            from astropy.convolution import convolve
            self.model_image = convolve(self.model_image, psf)

        random_state = get_random_state(random_state)

        # Fake measurements
        for _ in range(N):
            self.measurements.append(random_state.poisson(self.model_image))


def make_test_model(nsources=100, npix=500, ampl=100, fwhm=30,
                    random_state='random-seed'):
    """Create a model of several Gaussian sources.

    Parameters
    ----------
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`.
    """
    from sherpa.astro.ui import set_source
    from gammapy.image.models.utils import _set, _name

    # initialise random number generator
    random_state = get_random_state(random_state)

    model = ' + '.join([_name(ii) for ii in range(nsources)])
    set_source(model)
    for ii in range(nsources):
        _set(_name(ii), 'xpos', random_state.uniform(0, npix))
        _set(_name(ii), 'ypos', random_state.uniform(0, npix))
        _set(_name(ii), 'ampl', random_state.uniform(0, ampl))
        _set(_name(ii), 'fwhm', random_state.uniform(0, fwhm))


def read_json(filename):
    from sherpa.astro.ui import set_source
    from gammapy.image.models.utils import read_json
    read_json(filename, set_source)
