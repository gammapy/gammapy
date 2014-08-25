# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""An iterative multi-scale source detection method.

This is a prototype implementation of the following algorithm:
1. Input is: count, background and exposure map and list of scales
2. Compute significance maps on multiple scales (disk-correlate)
3. Largest peak on any scale gives a seed position / extension (the scale)
4. Fit a 2D Gauss-model source using the seed parameters
5. Add the source to a list of detected sources and the background model
6. Restart at 2, but this time with detected sources added to the background
   model, i.e. significance maps will be "residual significance" maps.

TODO: tons of things, e.g.
* Use Sherpa catalog pipeline for `fit_source_parameters step.
  This will automatically take care of these points:
    * Keep parameters of previously found sources free when adding a new source
    * Write more debug maps (e.g. excess)
      and info (e.g. sources_guess positions).
    * Add PSF convolution
* Use TS maps with Gauss source morphology instead of disk.
* Make it more modular and more abstract; put in gammapy.detect
  - user should be able to plug-in their significance map computation?
  - support different source models?
  - Separate Iterator, SignificanceMapCalculator, Guesser, Fitter ...
    and implement different methods as sub-classes or via callbacks?
  - e.g. list of peaks should be replaced with some abstract class that
    allows different characteristics / methods to be implemented.
* Introduce parameters that allow us to vary the procedure
* Check if Python garbage collection for iter_maps sets in OK
  or if explicit garbage collection is needed.
* Use photutils aperture photometry for estimate_flux?
* Introduce FLUX_SCALE = 1e-10 parameter to avoid roundoff error problems?
"""
from __future__ import print_function, division
import logging
import numpy as np
from astropy.io import fits
from .. import stats
from ..image import disk_correlate

__all__ = ['IterativeSourceDetector',
           'run_detection',
           ]


class FitFailedError(object):
    """Fit failed error.
    """
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

    def __init__(self, maps, scales, max_sources=10, significance_threshold=5,
                 max_ncall=300, debug_output_folder='', clobber=False):
        self.maps = maps
        # Note: FITS convention is to start counting pixels at 1
        y, x = np.indices(maps['counts'].shape, dtype=np.int32) + 1
        self.maps['x'], self.maps['y'] = x, y

        # Temp maps that change in each iteration
        self.iter_maps = dict()
        self.find_peaks = []

        self.scales = np.asanyarray(scales)
        self.max_sources = max_sources
        self.significance_threshold = significance_threshold
        self.max_ncall = max_ncall
        self.debug_output_folder = debug_output_folder
        self.clobber = clobber

        self.sources_guess = []
        self.sources = []

        # At the moment we only
        # self.peaks = np.zeros_like(self.scales)

    def run(self):
        """Run source detection."""
        logging.debug('Running source detection')

        for _ in range(self.max_sources):
            logging.debug('Starting iteration number {0}'.format(_))
            debug_folder = self.debug_output_folder + '/' + str(_)
            if self.debug_output_folder:
                try:
                    os.mkdir(debug_folder)
                    logging.info('mkdir {0}'.format(debug_folder))
                except:
                    logging.debug('Folder exists: {0}'.format(debug_folder))

            self.compute_iter_maps()
            if self.debug_output_folder:
                # Save per iteration maps
                for name in ['background']:
                    filename = '{0}/{1}.fits'.format(debug_folder, name)
                    logging.info('Writing {0}'.format(filename))
                    fits.writeto(filename, self.iter_maps[name], clobber=self.clobber)

                # Save per iteration and scale maps
                for name in ['significance']:
                    for scale in self.scales:
                        filename = '{0}/{1}_{2}.fits'.format(debug_folder, name, scale)
                        logging.info('Writing {0}'.format(filename))
                        fits.writeto(filename, self.iter_maps[name][scale], clobber=self.clobber)

            self.find_peaks()
            # TODO: debug output to JSON here and for later steps

            if self.stop_iteration():
                break

            self.guess_source_parameters()
            if self.debug_output_folder:
                filename = '{0}/{1}'.format(debug_folder, 'sources_guess.reg')
                self.save_regions(filename, selection='guess')

            try:
                self.fit_source_parameters()
            except FitFailedError:
                logging.warning('Fit failed. Full stop.')
                break

    def compute_iter_maps(self):
        """Compute maps for this iteration."""
        logging.debug('Computing maps for this iteration.')
        self.iter_maps = dict()

        background = self.maps['background']
        background += self.model_excess(self.sources)
        self.iter_maps['background'] = background

        self.iter_maps['significance'] = dict()
        for scale in self.scales:
            counts = disk_correlate(self.maps['counts'], scale)
            background = disk_correlate(self.iter_maps['background'], scale)
            significance = stats.significance(counts, background)
            self.iter_maps['significance'][scale] = significance

    def model_excess(self, sources):
        """Compute model excess image."""
        # logging.debug('Computing model excess')
        x, y = self.maps['x'], self.maps['y']
        flux = np.zeros_like(x, dtype=np.float64)
        for source in sources:
            # logging.debug('Adding source: {0}'.format(source))
            source_flux = gauss2d(x, y, **source)
            # logging.debug('Source flux: {0}'.format(source_flux.sum()))
            flux += source_flux
            # logging.debug('Total flux: {0}'.format(flux.sum()))
        excess = flux * self.maps['exposure']
        return excess

    def find_peaks(self):
        """Find peaks in residual significance image."""
        logging.debug('Finding peaks.')
        self.peaks = []
        for scale in self.scales:
            image = self.iter_maps['significance'][scale]
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
            logging.debug('Peak on scale {scale:5.2f} is at ({xpos:5d}, {ypos:5d}) with value {val:7.2f}'
                          ''.format(**peak))

    def stop_iteration(self):
        """Criteria to stop the iteration process."""
        max_significance = max([_['val'] for _ in self.peaks])
        if max_significance < self.significance_threshold:
            logging.debug('Max peak significance of {0:7.2f} is smaller than detection threshold {1:7.2f}'
                          ''.format(max_significance, self.significance_threshold))
            logging.debug('Stopping iteration.')
            return True
        else:
            return False

    def guess_source_parameters(self):
        """Guess source start parameters for the fit.

        At the moment take the position and scale of the maximum residual peak
        and compute the excess within a circle around that position.
        """
        logging.debug('Guessing Gauss source parameters:')

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
        logging.debug('xpos: {xpos}'.format(**source))
        logging.debug('ypos: {ypos}'.format(**source))
        logging.debug('sigma: {sigma}'.format(**source))
        source['flux'] = self.estimate_flux(source)
        self.sources_guess.append(source)

    def fit_source_parameters(self):
        """Fit source parameters using the guess as start values.

        For this prototype we simply roll our own using iminuit,
        this should probably be changed to astropy or Sherpa.
        """
        logging.debug('Fitting source parameters')
        from iminuit import Minuit

        def fit_stat(xpos, ypos, sigma, flux):
            """Define CASH fit statistic for Gauss model"""
            data = self.maps['counts']
            # Note: No need to re-compute excess model for all previous source,
            # that is already contained in the background in iter_maps.
            background = self.iter_maps['background']
            sources = [dict(xpos=xpos, ypos=ypos, sigma=sigma, flux=flux)]
            model = background + self.model_excess(sources)
            cash = stats.cash(data, model).sum()
            return cash

        source = self.sources_guess[-1]
        logging.debug('Source parameters before fit: {0}'.format(source))
        pars = source.copy()
        pars['error_xpos'] = 0.01
        pars['error_ypos'] = 0.01
        pars['error_flux'] = 0.1 * source['flux']
        pars['error_sigma'] = 0.1 * source['sigma']
        SIGMA_LIMITS = (0.01, 1e6)
        pars['limit_sigma'] = SIGMA_LIMITS
        # import IPython; IPython.embed(); 1 / 0
        minuit = Minuit(fit_stat, pedantic=False, print_level=1, **pars)
        # minuit.print_initial_param()
        minuit.migrad(ncall=self.max_ncall)

        source = minuit.values
        logging.debug('Source parameters  after fit: {0}'.format(source))

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
        logging.debug('Estimating flux')
        SOURCE_RADIUS_FACTOR = 2
        radius = SOURCE_RADIUS_FACTOR * source['sigma']
        r2 = ((self.maps['x'] - source['xpos']) ** 2 +
              (self.maps['y'] - source['ypos']) ** 2)
        mask = (r2 < radius ** 2)
        npix = mask.sum()
        if method == 'sum_and_divide':
            counts = self.maps['counts'][mask].sum()
            background = self.iter_maps['background'][mask].sum()
            # Note: exposure is not per pixel.
            # It has units m^2 s TeV
            exposure = self.maps['exposure'][mask].mean()
            excess = counts - background
            # TODO: check if true:
            # Flux is differential flux at 1 TeV in units m^-2 s^-1 TeV^-1
            # Or is it integral flux above 1 TeV in units of m^-2 s^-1?
            flux = excess / exposure
        elif method == 'divide_and_sum':
            counts = self.maps['counts'][mask].sum()
            background = self.iter_maps['background'][mask].sum()
            exposure = self.maps['exposure'][mask].mean()
            excess_image = self.maps['counts'] - self.iter_maps['background']
            excess = excess_image[mask].sum()
            flux_image = (self.maps['counts'] - self.iter_maps['background']) / self.maps['exposure']
            flux = flux_image[mask].sum()
        logging.debug('Flux estimation for source region radius: {0}'.format(radius))
        logging.debug('npix: {0}'.format(npix))
        logging.debug('counts: {0}'.format(counts))
        logging.debug('background: {0}'.format(background))
        logging.debug('excess: {0}'.format(excess))
        logging.debug('exposure: {0}'.format(exposure))
        logging.debug('flux: {0}'.format(flux))
        return flux

    def save_fits(self, filename):
        """Save source catalog to FITS file."""
        logging.info('Writing source detections in FITS format to {0}'.format(filename))
        # TODO

    def save_regions(self, filename, selection='fit'):
        """Save ds9 region file."""
        logging.info('Writing source detections in ds9 region format to {0}'.format(filename))
        if selection == 'fit':
            sources = self.sources
            color = 'green'
        elif selection == 'guess':
            sources = self.sources_guess
            color = 'magenta'
        else:
            raise ValueError('Unknown selection: {0}'.format(selection))
        with open(filename, 'w') as outfile:
            outfile.write('image\n')
            for ii, source in enumerate(sources):
                fmt = 'circle({xpos:3.3f},{ypos:3.3f},{radius:3.3f}) # text="{name}" color={color}\n'
                data = dict(xpos=source['xpos'], ypos=source['ypos'])
                N_SIGMA = 3
                data['radius'] = N_SIGMA * source['sigma']
                data['name'] = 'Source {0}'.format(ii)
                data['color'] = color
                text = fmt.format(**data)
                outfile.write(text)

    def save_json(self, filename):
        """Save source catalog to JSON file."""
        logging.info('Writing source detections in JSON format to {0}'.format(filename))
        import json
        data = dict(sources=self.sources, sources_guess=self.sources_guess)
        # print data
        # import IPython; IPython.embed(); 1/0
        # TODO: this fails because data contains np.float32 values, which are not JSON serializable:
        # TypeError: 1.2617354e-10 is not JSON serializable 
        with open(filename, 'w') as outfile:
            json.dump(data, outfile, indent=4)


def run_detection(args):
    """Run iterative source detection."""
    # Load data
    maps = dict()
    for mapname in ['counts', 'background', 'exposure']:
        filename = args[mapname]
        logging.info('Reading {0} map: {1}'.format(mapname, filename))
        maps[mapname] = fits.getdata(filename)

    # Compute scales in pixel coordinates
    DEG_PER_PIX = np.abs(fits.getval(args['counts'], 'CDELT1'))
    scales_deg = args['scales']
    scales_pix = np.array(scales_deg) / DEG_PER_PIX
    logging.info('Number of scales: {0}'.format(len(scales_deg)))
    logging.info('DEG_PER_PIX: {0}'.format(DEG_PER_PIX))
    logging.info('Scales in deg: {0}'.format(scales_deg))
    logging.info('Scales in pix: {0}'.format(scales_pix))

    # Run the iterative source detection
    detector = IterativeSourceDetector(maps=maps, scales=scales_pix,
                                       debug_output_folder=args['debug_output_folder'],
                                       clobber=True)
    detector.run()

    # Save the results
    # detector.save_fits(args['output_fits'])
    detector.save_regions(args['output_regions'])
    # detector.save_json('detect.json')
