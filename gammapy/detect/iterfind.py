# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""An iterative multi-scale source detection method.

This is a prototype implementation of the following algorithm:
1. Input is: count, background and exposure image and list of scales
2. Compute significance images on multiple scales (disk-correlate)
3. Largest peak on any scale gives a seed position / extension (the scale)
4. Fit a 2D Gauss-model source using the seed parameters
5. Add the source to a list of detected sources and the background model
6. Restart at 2, but this time with detected sources added to the background
   model, i.e. significance images will be "residual significance" images.

TODO: tons of things, e.g.
* Use Sherpa catalog pipeline for `fit_source_parameters step.
  This will automatically take care of these points:
    * Keep parameters of previously found sources free when adding a new source
    * Write more debug images (e.g. excess)
      and info (e.g. sources_guess positions).
    * Add PSF convolution
* Use TS images with Gauss source morphology instead of disk.
* Make it more modular and more abstract; put in gammapy.detect
  - user should be able to plug-in their significance image computation?
  - support different source models?
  - Separate Iterator, SignificanceImageCalculator, Guesser, Fitter ...
    and implement different methods as sub-classes or via callbacks?
  - e.g. list of peaks should be replaced with some abstract class that
    allows different characteristics / methods to be implemented.
* Introduce parameters that allow us to vary the procedure
* Check if Python garbage collection for iter_images sets in OK
  or if explicit garbage collection is needed.
* Use photutils aperture photometry for estimate_flux?
* Introduce FLUX_SCALE = 1e-10 parameter to avoid round-off error problems?
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.io import fits
from ..extern.pathlib import Path
from .. import stats

__all__ = [
    # TODO: not working, so not part of the docs yet
    # 'IterativeSourceDetector',
]

log = logging.getLogger(__name__)


class FitFailedError(object):
    """Fit failed error."""
    pass


def gauss2d(x, y, xpos, ypos, sigma, flux):
    """2D Gaussian source model."""
    x = np.asanyarray(x, dtype=np.float64)
    y = np.asanyarray(y, dtype=np.float64)
    theta2 = (x - xpos) ** 2 + (y - ypos) ** 2
    sigma2 = sigma * sigma
    term_a = 1 / (2 * np.pi * sigma2)
    term_b = np.exp(-0.5 * theta2 / sigma2)
    image = term_a * term_b
    return flux * image / image.sum()


class IterativeSourceDetector(object):
    """An iterative source detection algorithm.

    TODO: document

    Parameters
    ----------
    debug_output_folder : str
        Use empty string for no debug output.
    """

    def __init__(self, images, scales, max_sources=10, significance_threshold=5,
                 max_ncall=300, debug_output_folder='', overwrite=False):
        self.images = images
        # Note: FITS convention is to start counting pixels at 1
        y, x = np.indices(images['counts'].shape, dtype=np.int32) + 1
        self.images['x'], self.images['y'] = x, y

        # Temp images that change in each iteration
        self.iter_images = dict()
        self.find_peaks = []

        self.scales = np.asanyarray(scales)
        self.max_sources = max_sources
        self.significance_threshold = significance_threshold
        self.max_ncall = max_ncall
        self.debug_output_folder = debug_output_folder
        self.overwrite = overwrite

        self.sources_guess = []
        self.sources = []

        # At the moment we only
        # self.peaks = np.zeros_like(self.scales)

    def run(self):
        """Run source detection."""
        log.debug('Running source detection')

        for _ in range(self.max_sources):
            log.debug('Starting iteration number {}'.format(_))
            debug_folder = self.debug_output_folder + '/' + str(_)
            if self.debug_output_folder:
                Path(debug_folder).mkdir()
                log.info('mkdir {}'.format(debug_folder))

            self.compute_iter_images()
            if self.debug_output_folder:
                # Save per iteration images
                for name in ['background']:
                    filename = '{}/{}.fits'.format(debug_folder, name)
                    log.info('Writing {}'.format(filename))
                    fits.writeto(filename, self.iter_images[name], overwrite=self.overwrite)

                # Save per iteration and scale images
                for name in ['significance']:
                    for scale in self.scales:
                        filename = '{}/{}_{}.fits'.format(debug_folder, name, scale)
                        log.info('Writing {}'.format(filename))
                        fits.writeto(filename, self.iter_images[name][scale], overwrite=self.overwrite)

            self.find_peaks()
            # TODO: debug output to JSON here and for later steps

            if self.stop_iteration():
                break

            self.guess_source_parameters()
            if self.debug_output_folder:
                filename = '{}/{}'.format(debug_folder, 'sources_guess.reg')
                self.save_regions(filename, selection='guess')

            try:
                self.fit_source_parameters()
            except FitFailedError:
                log.warning('Fit failed. Full stop.')
                break

    def compute_iter_images(self):
        """Compute images for this iteration."""
        from scipy.ndimage import convolve
        log.debug('Computing images for this iteration.')
        self.iter_images = dict()

        background = self.images['background']
        background += self.model_excess(self.sources)
        self.iter_images['background'] = background

        self.iter_images['significance'] = dict()
        for scale in self.scales:
            disk = Tophat2Dkernel(scale)
            disk.normalize('peak')
            counts = convolve(self.images['counts'], disk.array)
            background = convolve(self.iter_images['background'], disk.array)
            significance = stats.significance(counts, background)
            self.iter_images['significance'][scale] = significance

    def model_excess(self, sources):
        """Compute model excess image."""
        x, y = self.images['x'], self.images['y']
        flux = np.zeros_like(x, dtype=np.float64)
        for source in sources:
            source_flux = gauss2d(x, y, **source)
            flux += source_flux
        excess = flux * self.images['exposure']
        return excess

    def find_peaks(self):
        """Find peaks in residual significance image."""
        log.debug('Finding peaks.')
        self.peaks = []
        for scale in self.scales:
            image = self.iter_images['significance'][scale]
            # Note: significance images sometimes contain Inf or NaN values.
            # We set them here to a value so that they will be ignored
            mask = np.invert(np.isfinite(image))
            image[mask] = -1e10

            # This is how to find the peak position in a 2D numpy array
            y, x = np.unravel_index(np.nanargmax(image), image.shape)
            val = image[y, x]
            peak = dict()
            peak['xpos'], peak['ypos'] = x, y
            peak['val'], peak['scale'] = val, scale
            self.peaks.append(peak)
            log.debug('Peak on scale {scale:5.2f} is at ({xpos:5d}, {ypos:5d}) with value {val:7.2f}'
                      ''.format(**peak))

    def stop_iteration(self):
        """Criteria to stop the iteration process."""
        max_significance = max([_['val'] for _ in self.peaks])
        if max_significance < self.significance_threshold:
            log.debug('Max peak significance of {0:7.2f} is smaller than detection threshold {1:7.2f}'
                      ''.format(max_significance, self.significance_threshold))
            log.debug('Stopping iteration.')
            return True
        else:
            return False

    def guess_source_parameters(self):
        """Guess source start parameters for the fit.

        At the moment take the position and scale of the maximum residual peak
        and compute the excess within a circle around that position.
        """
        log.debug('Guessing Gauss source parameters:')

        # Find the scale with the most significant peak
        peak = self.peaks[0]
        for _ in range(1, len(self.scales)):
            if self.peaks[_]['val'] > peak['val']:
                peak = self.peaks[_]

        source = dict()
        source['xpos'], source['ypos'] = peak['xpos'], peak['ypos']
        # TODO: introduce rough scale factor disk -> gauss here
        SIGMA_SCALE_FACTOR = 1
        source['sigma'] = SIGMA_SCALE_FACTOR * peak['scale']
        log.debug('xpos: {xpos}'.format(**source))
        log.debug('ypos: {ypos}'.format(**source))
        log.debug('sigma: {sigma}'.format(**source))
        source['flux'] = self.estimate_flux(source)
        self.sources_guess.append(source)

    def fit_source_parameters(self):
        """Fit source parameters using the guess as start values.

        For this prototype we simply roll our own using iminuit,
        this should probably be changed to astropy or Sherpa.
        """
        log.debug('Fitting source parameters')
        from iminuit import Minuit

        def fit_stat(xpos, ypos, sigma, flux):
            """Define CASH fit statistic for Gauss model"""
            data = self.images['counts']
            # Note: No need to re-compute excess model for all previous source,
            # that is already contained in the background in iter_images.
            background = self.iter_images['background']
            sources = [dict(xpos=xpos, ypos=ypos, sigma=sigma, flux=flux)]
            model = background + self.model_excess(sources)
            cash = stats.cash(data, model).sum()
            return cash

        source = self.sources_guess[-1]
        log.debug('Source parameters before fit: {}'.format(source))
        pars = source.copy()
        pars['error_xpos'] = 0.01
        pars['error_ypos'] = 0.01
        pars['error_flux'] = 0.1 * source['flux']
        pars['error_sigma'] = 0.1 * source['sigma']
        SIGMA_LIMITS = (0.01, 1e6)
        pars['limit_sigma'] = SIGMA_LIMITS
        minuit = Minuit(fit_stat, pedantic=False, print_level=1, **pars)
        # minuit.print_initial_param()
        minuit.migrad(ncall=self.max_ncall)

        source = minuit.values
        log.debug('Source parameters  after fit: {}'.format(source))

        if not minuit.migrad_ok():
            # If fit doesn't converge we simply abort
            # TODO: should we use exceptions here or return False as signal?
            minuit.print_fmin()
            raise FitFailedError
        else:
            # Store best-fit source parameters
            self.sources.append(source)

    def estimate_flux(self, source, method='sum_and_divide'):
        """Estimate flux in a circular region around the source.

        Note: It's not clear which is the better flux estimate.

        * ``method == 'sum_and_divide'``::

              flux = (counts.sum() - background.sum()) / exposure.mean()

        * ``method = 'divide_and_sum'``::

              flux = ((counts - background) / exposure).sum()
        """
        log.debug('Estimating flux')
        SOURCE_RADIUS_FACTOR = 2
        radius = SOURCE_RADIUS_FACTOR * source['sigma']
        r2 = ((self.images['x'] - source['xpos']) ** 2 +
              (self.images['y'] - source['ypos']) ** 2)
        mask = (r2 < radius ** 2)
        npix = mask.sum()
        if method == 'sum_and_divide':
            counts = self.images['counts'][mask].sum()
            background = self.iter_images['background'][mask].sum()
            # Note: exposure is not per pixel.
            # It has units m^2 s TeV
            exposure = self.images['exposure'][mask].mean()
            excess = counts - background
            # TODO: check if true:
            # Flux is differential flux at 1 TeV in units m^-2 s^-1 TeV^-1
            # Or is it integral flux above 1 TeV in units of m^-2 s^-1?
            flux = excess / exposure
        elif method == 'divide_and_sum':
            counts = self.images['counts'][mask].sum()
            background = self.iter_images['background'][mask].sum()
            exposure = self.images['exposure'][mask].mean()
            excess_image = self.images['counts'] - self.iter_images['background']
            excess = excess_image[mask].sum()
            flux_image = (self.images['counts'] - self.iter_images['background']) / self.images['exposure']
            flux = flux_image[mask].sum()
        log.debug('Flux estimation for source region radius: {}'.format(radius))
        log.debug('npix: {}'.format(npix))
        log.debug('counts: {}'.format(counts))
        log.debug('background: {}'.format(background))
        log.debug('excess: {}'.format(excess))
        log.debug('exposure: {}'.format(exposure))
        log.debug('flux: {}'.format(flux))
        return flux

    def save_fits(self, filename):
        """Save source catalog to FITS file."""
        log.info('Writing source detections in FITS format to {}'.format(filename))
        # TODO

    def save_regions(self, filename, selection='fit'):
        """Save ds9 region file."""
        log.info('Writing source detections in ds9 region format to {}'.format(filename))
        if selection == 'fit':
            sources = self.sources
            color = 'green'
        elif selection == 'guess':
            sources = self.sources_guess
            color = 'magenta'
        else:
            raise ValueError('Unknown selection: {}'.format(selection))
        with open(filename, 'w') as outfile:
            outfile.write('image\n')
            for ii, source in enumerate(sources):
                fmt = 'circle({xpos:3.3f},{ypos:3.3f},{radius:3.3f}) # text="{name}" color={color}\n'
                data = dict(xpos=source['xpos'], ypos=source['ypos'])
                N_SIGMA = 3
                data['radius'] = N_SIGMA * source['sigma']
                data['name'] = 'Source {}'.format(ii)
                data['color'] = color
                text = fmt.format(**data)
                outfile.write(text)

    def save_json(self, filename):
        """Save source catalog to JSON file."""
        log.info('Writing source detections in JSON format to {}'.format(filename))
        import json
        data = dict(sources=self.sources, sources_guess=self.sources_guess)
        # TODO: this fails because data contains np.float32 values, which are not JSON serializable:
        # TypeError: 1.2617354e-10 is not JSON serializable
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)
