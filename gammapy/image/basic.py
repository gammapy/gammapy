# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Basic sky image estimator classes
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict

import numpy as np
from astropy import units as u

from .core import SkyImage, SkyImageList
from ..cube import SkyCube, compute_npred_cube
from ..spectrum.models import PowerLaw2

SPECTRAL_INDEX = 2.3


class FermiLATBasicImageEstimator(object):
    """
    Make basic (counts, exposure and background) Fermi sky images in given
    energy band.

    TODO: allow different background estimation methods

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
    """

    def __init__(self, reference, emin, emax, spectral_model=None):
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

    def _counts_image(self, dataset):
        """
        Compute counts image in energy band
        """
        p = self.parameters
        events = dataset.events.select_energy((p['emin'], p['emax']))

        counts = self._get_empty_skyimage('counts')
        counts.fill_events(events)
        return counts

    #TODO: move this method to a separate GalacticDiffuseBackgroundEstimator?
    def _background_image(self, dataset):
        """
        Compute predicted counts background image in energy band.
        """
        p = self.parameters
        erange = u.Quantity([p['emin'], p['emax']])

        exposure_cube = dataset.exposure

        background_total = dataset.galactic_diffuse.reproject(exposure_cube)

        # evaluate and add isotropic diffuse model
        energies = background_total.energies()
        flux = dataset.isotropic_diffuse(energies)
        background_total.data += flux.reshape(-1, 1, 1)

        # compute npred cube
        npred_cube = compute_npred_cube(background_total, exposure_cube, energy_bins=erange)

        # extract the only image from the npred_cube
        npred_total = npred_cube.sky_image_idx(0)

        psf_mean = dataset.psf.table_psf_in_energy_band(erange, spectrum=self.spectral_model)

        # convolve with PSF kernel
        kernel = psf_mean.kernel(npred_total)
        npred_total = npred_total.convolve(kernel)

        # reproject to reference image and renormalize
        norm = (npred_total.wcs_pixel_scale() / self.reference.wcs_pixel_scale())
        npred_total = npred_total.reproject(self.reference)

        npred_total.data /= (norm.mean()) ** 2
        return npred_total

    def _exposure_image(self, dataset):
        """
        Compute a powerlaw weighted exposure image from an exposure cube.
        """
        p = self.parameters

        exposure_cube = dataset.exposure
        exposure_weighted = SkyCube.empty_like(exposure_cube)
        energies = exposure_weighted.energies('center')

        weights = self.spectral_model(energies)
        exposure_weighted.data = exposure_cube.data * weights.reshape(-1, 1, 1)

        exposure = exposure_weighted.sky_image_integral(emin=p['emin'], emax=p['emax'])
        exposure.data = exposure.data.to('cm2 s').value
        exposure.unit = u.Unit('cm2 s')
        exposure.name = 'exposure'
        return exposure.reproject(self.reference)

    def _psf_image(self, dataset, nxpix=101, nypix=101, binsz=0.02):
        """
        Compute fermi PSF image.
        """
        p = self.parameters
        psf = dataset.psf

        erange = u.Quantity((p['emin'], p['emax']))
        psf_mean = psf.table_psf_in_energy_band(erange, spectrum=self.spectral_model)

        psf_image = SkyImage.empty(nxpix=nxpix, nypix=nypix, binsz=binsz)

        coordinates = psf_image.coordinates()
        offset = coordinates.separation(psf_image.center)
        psf_image.data = psf_mean.evaluate(offset)

        # normalize PSF
        psf_image.data = (psf_image.data / psf_image.data.sum()).value
        return psf_image

    def run(self, dataset):
        """
        Make sky images.

        The following images will be computed:

            * counts
            * background (predicted counts)
            * exposure
            * excess
            * flux
            * psf

        Parameters
        ----------
        dataset : `FermiDataset`
            Fermi basic dataset to compute images for.

        Returns
        -------
        images : `~gammapy.images.SkyImageList`
            List of sky images.
        """
        images = SkyImageList()
        counts = self._counts_image(dataset)
        images['counts'] = counts

        background = self._background_image(dataset)
        images['background'] = background

        exposure = self._exposure_image(dataset)
        images['exposure'] = exposure

        images['excess'] = self._get_empty_skyimage('excess')
        images['excess'].data = counts.data - background.data

        images['flux'] = self._get_empty_skyimage('flux')
        images['flux'].data = images['excess'].data / exposure.data

        images['psf'] = self._psf_image(dataset)
        return images
