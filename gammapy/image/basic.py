# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Basic sky image estimator classes
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict

import numpy as np
from astropy import units as u

from .core import SkyImage
from .lists import SkyImageList

__all__ = ['FermiLATBasicImageEstimator']

SPECTRAL_INDEX = 2.3


class FermiLATBasicImageEstimator(object):
    """
    Make basic (counts, exposure and background) Fermi sky images in given
    energy band.

    TODO: allow different background estimation methods
    TODO: add examples

    Parameters
    ----------
    reference : `~gammapy.image.SkyImage`
        Reference sky image.
    emin : `~astropy.units.Quantity`
        Lower bound of energy range.
    emax : `~astropy.units.Quantity`
        Upper bound of energy range.
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model assumption to compute mean exposure and psf images.

    Examples
    --------
    This example shows how to compute a set of basic images for the galactic
    center region using a prepared 2FHL dataset:

    .. code::

        from astropy import unit as u
        from gammapy.image import SkyImage, FermiLATBasicImageEstimator
        from gammapy.datasets import FermiLATDataset

        kwargs = {}
        kwargs['reference'] = SkyImage.empty(nxpix=201, nypix=101, binsz=0.05)
        kwargs['emin'] = 50 * u.GeV
        kwargs['emax'] = 3000 * u.GeV
        image_estimator = FermiLATBasicImageEstimator(**kwargs)

        filename = '$FERMI_LAT_DATA/2fhl/fermi_2fhl_data_config.yaml'
        dataset = FermiLATDataset(filename)

        result = image_estimator.run(dataset)
        result['counts'].show()

    """

    def __init__(self, reference, emin, emax, spectral_model=None):
        from ..spectrum.models import PowerLaw2

        self.reference = reference

        if spectral_model is None:
            index = SPECTRAL_INDEX
            amplitude = u.Quantity(1, '')
            spectral_model = PowerLaw2(index=index, amplitude=amplitude,
                                       emin=emin, emax=emax)

        self.spectral_model = spectral_model
        self.parameters = OrderedDict(emin=emin, emax=emax)

    def _get_empty_skyimage(self, name):
        """
        Get empty sky image like reference image.
        """
        p = self.parameters
        image = SkyImage.empty_like(self.reference)
        image.meta['emin'] = str(p['emin'])
        image.meta['emax'] = str(p['emax'])
        image.name
        return image

    def counts(self, dataset):
        """
        Estimate counts image in energy band

        Parameters
        ----------
        dataset : `~gammapy.datasets.FermiLATDataset`
            Fermi basic dataset to compute images for.

        Returns
        -------
        counts : `~gammapy.images.SkyImage`
            Counts sky image.
        """
        p = self.parameters
        events = dataset.events.select_energy((p['emin'], p['emax']))

        counts = self._get_empty_skyimage('counts')
        counts.fill_events(events)
        return counts

    def _cutout_background_cube(self, dataset):
        """
        Cutout reference region from galactic diffuse background model.

        Parameters
        ----------
        dataset : `~gammapy.datasets.FermiLATDataset`
            Fermi basic dataset to compute images for.

        Returns
        -------
        counts : `~gammapy.images.SkyImage`
            Counts sky image.
        """
        galactic_diffuse = dataset.galactic_diffuse

        # add margin of 1 pixel
        margin = galactic_diffuse.sky_image_ref.wcs_pixel_scale()

        footprint = self.reference.footprint(mode='edges')
        width = footprint['width'] + margin[1]
        height = footprint['height'] + margin[0]

        cutout = galactic_diffuse.cutout(position=self.reference.center,
                                         size=(height, width))
        return cutout

    def _total_background_cube(self, dataset):
        """
        Compute total background compute for reference region.

        Parameters
        ----------
        dataset : `~gammapy.datasets.FermiLATDataset`
            Fermi basic dataset to compute images for.

        Returns
        -------
        counts : `~gammapy.images.SkyImage`
            Counts sky image.
        """
        background_total = self._cutout_background_cube(dataset)

        # evaluate and add isotropic diffuse model
        energies = background_total.energies()
        flux = dataset.isotropic_diffuse(energies)
        background_total.data += flux.reshape(-1, 1, 1)
        return background_total

    #TODO: move this method to a separate GalacticDiffuseBackgroundEstimator?
    def background(self, dataset):
        """
        Estimate predicted counts background image in energy band.

        The background estimati is based on the Fermi-LAT galactic and
        isotropic diffuse models.

        Parameters
        ----------
        dataset : `~gammapy.datasets.FermiLATDataset`
            Fermi basic dataset to compute images for.

        Returns
        -------
        background : `~gammapy.images.SkyImage`
            Predicated number of background counts sky image.
        """
        from ..cube import compute_npred_cube

        p = self.parameters
        erange = u.Quantity([p['emin'], p['emax']])

        background_cube = self._total_background_cube(dataset)
        exposure_cube = dataset.exposure.reproject(background_cube)
        psf = dataset.psf

        # compute npred cube
        npred_cube = compute_npred_cube(background_cube, exposure_cube, energy_bins=erange)

        # extract the only image from the npred_cube
        npred_total = npred_cube.sky_image_idx(0)

        # reproject to reference image and renormalize data
        # TODO: use solid angle image
        norm = (npred_total.wcs_pixel_scale() / self.reference.wcs_pixel_scale())
        npred_total = npred_total.reproject(self.reference)
        npred_total.data /= (norm.mean()) ** 2

        # convolve with PSF kernel
        psf_mean = psf.table_psf_in_energy_band(erange, spectrum=self.spectral_model)
        kernel = psf_mean.kernel(npred_total)
        npred_total = npred_total.convolve(kernel)
        return npred_total

    def exposure(self, dataset):
        """
        Estimate a spectral model weighted exposure image from an exposure cube.

        Parameters
        ----------
        dataset : `~gammapy.datasets.FermiLATDataset`
            Fermi basic dataset to compute images for.

        Returns
        -------
        exposure : `~gammapy.images.SkyImage`
            Exposure sky image.
        """
        from ..cube import SkyCube

        p = self.parameters

        ref_cube = self._cutout_background_cube(dataset)
        exposure_cube = dataset.exposure.reproject(ref_cube)

        exposure_weighted = SkyCube.empty_like(exposure_cube)
        energies = exposure_weighted.energies('center')

        weights = self.spectral_model(energies)
        exposure_weighted.data = exposure_cube.data * weights.reshape(-1, 1, 1)

        exposure = exposure_weighted.sky_image_integral(emin=p['emin'], emax=p['emax'])
        # TODO: check why fixing the unit is needed
        exposure.data = exposure.data.to('cm2 s').value
        exposure.unit = u.Unit('cm2 s')
        exposure.name = 'exposure'
        return exposure.reproject(self.reference)

    def _psf_image(self, dataset, nxpix=101, nypix=101, binsz=0.02):
        """
        Compute fermi PSF image.
        """
        p = self.parameters
        psf_image = SkyImage.empty(nxpix=nxpix, nypix=nypix, binsz=binsz)

        psf = dataset.psf
        erange = u.Quantity((p['emin'], p['emax']))
        psf_mean = psf.table_psf_in_energy_band(erange, spectrum=self.spectral_model)

        coordinates = psf_image.coordinates()
        offset = coordinates.separation(psf_image.center)
        psf_image.data = psf_mean.evaluate(offset)

        # normalize PSF
        psf_image.data = (psf_image.data / psf_image.data.sum()).value
        return psf_image

    @staticmethod
    def excess(images):
        """
        Estimate excess image.

        Requires 'counts' and 'background' image.

        Parameters
        ----------
        images : `~gammapy.images.SkyImageList`
            List of sky images.

        Returns
        -------
        excess : `~gammapy.images.SkyImage`
            Excess sky image.
        """
        images.check_required(['counts', 'background'])
        excess = SkyImage.empty_like(images['counts'], name='excess')
        excess.data = images['counts'].data - images['background'].data
        return excess

    @staticmethod
    def flux(images):
        """
        Estimate flux image.

        Requires 'counts', 'background' and 'exposure' image.

        Parameters
        ----------
        images : `~gammapy.images.SkyImageList`
            List of sky images.

        Returns
        -------
        flux : `~gammapy.images.SkyImage`
            Flux sky image.
        """
        #TODO: differentiate between flux (integral flux) and dnde (differential flux)
        images.check_required(['counts', 'background', 'exposure'])
        flux = SkyImage.empty_like(images['counts'], name='flux')
        excess = images['counts'].data - images['background'].data
        flux.data = excess / images['exposure']
        return flux

    def run(self, dataset, which='all'):
        """
        Estimate sky images.

        Parameters
        ----------
        dataset : `~gammapy.datasets.FermiLATDataset`
            Fermi basic dataset to compute images for.
        which : str or list of str
            Which images to compute. Can be:

                * 'all'
                * 'counts'
                * 'background'
                * 'exposure'
                * 'excess'
                * 'flux'
                * 'psf'

            Or a list containing any subset of the images listed above.

        Returns
        -------
        images : `~gammapy.images.SkyImageList`
            List of sky images.
        """
        images = SkyImageList()
        which = np.atleast_1d(which)

        if 'all' in which:
            which = ['counts', 'exposure', 'background', 'excess', 'flux', 'psf']

        if 'counts' in which:
            images['counts'] = self.counts(dataset)

        if 'background' in which:
            images['background'] = self.background(dataset)

        if 'exposure' in which:
            images['exposure'] = self.exposure(dataset)

        if 'excess' in which:
            images['excess'] = self.excess(images)

        if 'flux' in which:
            images['flux'] = self.flux(images)

        if 'psf' in which:
            images['psf'] = self._psf_image(dataset)
        return images
