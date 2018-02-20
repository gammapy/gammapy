# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Basic sky image estimator classes
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.wcs import WCS
from ..utils.energy import Energy
from .core import SkyImage
from .lists import SkyImageList

__all__ = [
    'BasicImageEstimator',
    'IACTBasicImageEstimator',
    'FermiLATBasicImageEstimator',
]

SPECTRAL_INDEX = 2.3


class BasicImageEstimator(object):
    """
    BasicImageEstimator base class.
    """

    @property
    def _default_spectral_model(self):
        p = self.parameters
        from ..spectrum.models import PowerLaw2
        index = SPECTRAL_INDEX
        amplitude = u.Quantity(1, '')
        return PowerLaw2(index=index, amplitude=amplitude,
                         emin=p['emin'], emax=p['emax'])

    def _get_empty_skyimage(self, name):
        """
        Get empty sky image like reference image.
        """
        p = self.parameters
        image = SkyImage.empty_like(self.reference)
        image.meta['emin'] = str(p['emin'])
        image.meta['emax'] = str(p['emax'])
        image.name = name
        return image

    @staticmethod
    def excess(images):
        """
        Estimate excess image.

        Requires 'counts' and 'background' image.

        Parameters
        ----------
        images : `~gammapy.image.SkyImageList`
            List of sky images.

        Returns
        -------
        excess : `~gammapy.image.SkyImage`
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
        images : `~gammapy.image.SkyImageList`
            List of sky images.

        Returns
        -------
        flux : `~gammapy.image.SkyImage`
            Flux sky image.
        """
        # TODO: differentiate between flux (integral flux) and dnde (differential flux)
        images.check_required(['counts', 'background', 'exposure'])
        flux = SkyImage.empty_like(images['counts'], name='flux')
        excess = images['counts'].data - images['background'].data

        with np.errstate(invalid='ignore', divide='ignore'):
            flux.data = (excess / images['exposure'].data)
        flux.unit = (1 / images['exposure'].unit).unit

        is_zero = images['exposure'].data == 0
        flux.data[is_zero] = 0
        return flux


class IACTBasicImageEstimator(BasicImageEstimator):
    """
    Estimate the basic sky images for a set of IACT observations.

    The following images will be computed:

    * counts
    * exposure
    * background

    Parameters
    ----------
    reference : `~gammapy.image.SkyImage`
        Reference sky image.
    emin : `~astropy.units.Quantity`
        Lower bound of energy range.
    emax : `~astropy.units.Quantity`
        Upper bound of energy range.
    offset_max : `~astropy.coordinates.Angle`
        Upper bound of offset range.
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model assumption to compute mean exposure and psf image.
    background_estimator : TODO
        Instance of background estimation method.
    exclusion_mask : `~gammapy.image.SkyImage`
        Exclusion mask.
    """

    def __init__(self, reference, emin, emax, offset_max=Angle(2.5, 'deg'), spectral_model=None,
                 background_estimator=None, exclusion_mask=None):
        self.parameters = OrderedDict(emin=emin, emax=emax, offset_max=offset_max)
        self.reference = reference
        self.reference.data = np.zeros(reference.data.shape)
        self.background_estimator = background_estimator
        self.exclusion_mask = exclusion_mask

        if spectral_model is None:
            spectral_model = self._default_spectral_model

        self.spectral_model = spectral_model

    def _get_ref_cube(self, observation, enumbins=11):
        from ..cube import SkyCube
        from ..spectrum import LogEnergyAxis

        p = self.parameters

        cutout = self._cutout_observation(self.reference, observation)

        wcs = cutout.wcs.deepcopy()
        shape = (enumbins,) + cutout.data.shape
        data = np.zeros(shape)

        energy = Energy.equal_log_spacing(p['emin'], p['emax'], enumbins, 'TeV')
        energy_axis = LogEnergyAxis(energy, mode='center')

        return SkyCube(data=data, wcs=wcs, energy_axis=energy_axis)

    def _cutout_observation(self, image, observation, margin=0.1 * u.deg):
        p = self.parameters
        position = observation.pointing_radec
        size = 2 * (p['offset_max'] + margin)
        cutout = image.cutout(position=position, size=size, copy=False)
        return cutout

    def _exposure_cube(self, observation):
        """
        Estimate exposure cube for one observation.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            Observation object
        """
        from ..cube import make_exposure_cube
        p = self.parameters
        return make_exposure_cube(
            livetime=observation.observation_live_time_duration,
            pointing=observation.pointing_radec,
            aeff=observation.aeff,
            offset_max=p['offset_max'],
            ref_cube=self._get_ref_cube(observation),
        )

    def _exposure(self, observation):
        p = self.parameters
        exposure_cube = self._exposure_cube(observation)

        energies = exposure_cube.energies('center')
        weights = self.spectral_model(energies)

        exposure_cube.data = exposure_cube.data * weights.reshape(-1, 1, 1)
        exposure = exposure_cube.sky_image_integral(emin=p['emin'], emax=p['emax'])

        exposure.name = 'exposure'
        exposure.unit = exposure.data.unit
        exposure.data = np.nan_to_num(exposure.data.value)
        return exposure

    def psf(self, observations, containment_fraction = 0.99, rad_max = None):
        """Mean point spread function kernel image.

        Parameters
        ----------
        observations : `~gammapy.data.ObservationList`
            List of observations
        containment_fraction : float (0.99)
            Minimum PSF containment fraction included in kernel image.
        rad_max : `~astropy.coordinates.Angle` (None)
            If specified, passed to `~gammapy.irf.TablePSF.kernel`;
            containment_fraction is then ignored.

        Returns
        -------
        kernel : `~gammapy.image.SkyImage`
            PSF kernel as sky image.
        """
        p = self.parameters

        refskyim  = self.reference
        refskypos = refskyim.center
        mean_psf = observations.make_mean_psf(refskypos)

        erange = u.Quantity((p['emin'], p['emax']))
        psf_mean = mean_psf.table_psf_in_energy_band(erange, spectrum=self.spectral_model)

        if rad_max is None:
            rad_max = psf_mean.containment_radius(containment_fraction)
        else:
            rad_max = Angle(rad_max)

        psfkern = psf_mean.kernel(refskyim, rad_max)
        psfunit = psfkern.unit
        psfdata = psfkern.value

        psfhead = refskyim.wcs.to_header()
        radnpix = int(np.shape(psfdata)[0] / 2)
        psfhead['CRPIX1'] = radnpix + 1.0
        psfhead['CRPIX2'] = radnpix + 1.0
        psfwcs = WCS(psfhead)

        psf_image = SkyImage('psf', data=psfdata, wcs=psfwcs, unit=psfunit)
        return psf_image

    def _counts(self, observation):
        """
        Estimate counts image for one observation.

        Parameters
        ----------
        observation : `~gammapy.data.DataStoreObservation`
            Observation object
        """
        p = self.parameters
        events = observation.events.select_energy((p['emin'], p['emax']))

        counts = self._cutout_observation(self.reference, observation)
        counts.name = 'counts'
        counts.fill_events(events)
        return counts

    def _background(self, counts, exposure, observation):
        """
        Estimate background image for one observation.

        Parameters
        ----------
        TODO
        """
        input_images = SkyImageList()
        input_images['counts'] = counts

        # TODO: instead of using a constant exposure, the acceptance model should
        # be taken into account
        exposure_on = exposure.copy()
        exposure_on.name = 'exposure_on'
        exposure_on.data = (exposure_on.data > 0).astype(float)
        input_images['exposure_on'] = exposure_on
        input_images['exclusion'] = self._cutout_observation(self.exclusion_mask, observation)
        return self.background_estimator.run(input_images)

    def run(self, observations, which='all'):
        """
        Run IACT basic image estimation for a list of observations.

        Parameters
        ----------
        observations : `~gammapy.data.ObservationList`
            List of observations

        Returns
        -------
        sky_images : `~gammapy.image.SkyImageList`
            List of sky images
        """
        from astropy.utils.console import ProgressBar
        result = SkyImageList()

        if 'all' in which:
            which = ['counts', 'exposure', 'background', 'excess', 'flux', 'psf']

        for name in which:
            result[name] = self._get_empty_skyimage(name)

        for observation in ProgressBar(observations):
            if 'exposure' in which:
                exposure = self._exposure(observation)
                result['exposure'].paste(exposure)
                # TODO: improve SkyImage.paste() so that it enforces compatibility
                # of units when doing the sum. The fix below can then be removed.
                result['exposure'].unit = exposure.unit

            if 'counts' in which:
                counts = self._counts(observation)
                # TODO: on the left side of the field of view there is one extra
                # row of pixels in the counts image compared to the exposure and
                # background image. Check why this happends and remove the fix below
                not_has_exposure = ~(exposure.data > 0)
                counts.data[not_has_exposure] = 0
                result['counts'].paste(counts)

            if 'background' in which:
                background = self._background(counts, exposure, observation)['background']

                # TODO: include stacked alpha and on/off exposure images
                result['background'].paste(background)

            if 'excess' in which:
                excess = self.excess(SkyImageList([counts, background]))
                result['excess'].paste(excess)

            if 'flux' in which:
                flux = self.flux(SkyImageList([counts, background, exposure]))
                result['flux'].paste(flux)
                # TODO: improve SkyImage.paste() so that it enforces compatibility
                # of units when doing the sum. The fix below can then be removed.
                result['flux'].unit = flux.unit

        if 'psf' in which:
            result['psf'] = self.psf(observations)
        return result


class FermiLATBasicImageEstimator(BasicImageEstimator):
    """Estimate basic sky images for Fermi-LAT data.

    Can compute the following images: counts, exposure, background

    TODO: allow different background estimation methods
    TODO: add examples

    Parameters
    ----------
    reference : `~gammapy.image.SkyImage`
        Reference sky image
    emin, emax : `~astropy.units.Quantity`
        Energy range
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model assumption to compute mean exposure and psf images.
    rad_max : `~astropy.coordinates.Angle`
        PSF kernel size, passed to :func:`gammapy.irf.TablePSF.kernel`

    Examples
    --------
    This example shows how to compute a set of basic images for the galactic
    center region using a prepared 2FHL dataset:

    .. code:: python

        from astropy import units as u
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

    def __init__(self, reference, emin, emax, spectral_model=None,
                 rad_max=1 * u.deg):
        self.parameters = OrderedDict(emin=emin, emax=emax, rad_max=rad_max)
        self.reference = reference
        if spectral_model is None:
            self.spectral_model = self._default_spectral_model

    def counts(self, dataset):
        """
        Estimate counts image in energy band.

        Parameters
        ----------
        dataset : `~gammapy.datasets.FermiLATDataset`
            Fermi basic dataset to compute images for.

        Returns
        -------
        counts : `~gammapy.image.SkyImage`
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
        counts : `~gammapy.image.SkyImage`
            Counts sky image.
        """
        galactic_diffuse = dataset.galactic_diffuse

        margin = galactic_diffuse.sky_image_ref.wcs_pixel_scale()
        width = self.reference.width + margin[1]
        height = self.reference.height + margin[0]
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
        counts : `~gammapy.image.SkyImage`
            Counts sky image.
        """
        background_total = self._cutout_background_cube(dataset)

        # evaluate and add isotropic diffuse model
        energies = background_total.energies()
        flux = dataset.isotropic_diffuse(energies)
        background_total.data += flux.reshape(-1, 1, 1)
        return background_total

    # TODO: move this method to a separate GalacticDiffuseBackgroundEstimator?
    def background(self, dataset):
        """
        Estimate predicted counts background image in energy band.

        The background estimate is based on the Fermi-LAT galactic and
        isotropic diffuse models.

        Parameters
        ----------
        dataset : `~gammapy.datasets.FermiLATDataset`
            Fermi basic dataset to compute images for.

        Returns
        -------
        background : `~gammapy.image.SkyImage`
            Predicted number of background counts sky image.
        """
        from ..cube import compute_npred_cube

        p = self.parameters
        energy_band = u.Quantity([p['emin'], p['emax']])

        background_cube = self._total_background_cube(dataset)

        exposure_cube = reproject_exposure(dataset.exposure, background_cube)

        # compute npred cube
        npred_cube = compute_npred_cube(background_cube, exposure_cube, ebounds=energy_band)

        # extract the only image from the npred_cube
        npred_total = npred_cube.sky_image_idx(0)

        # reproject to reference image and renormalize data
        # TODO: use solid angle image
        norm = (npred_total.wcs_pixel_scale() / self.reference.wcs_pixel_scale()).to('')
        npred_total = npred_total.reproject(self.reference)
        npred_total.data /= (norm.value.mean()) ** 2

        # convolve with PSF kernel
        psf_mean = dataset.psf.table_psf_in_energy_band(energy_band, spectrum=self.spectral_model)
        kernel = psf_mean.kernel(npred_total, rad_max=self.parameters['rad_max'])
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
        exposure : `~gammapy.image.SkyImage`
            Exposure sky image.
        """
        from ..catalog.gammacat import NoDataAvailableError

        p = self.parameters

        try:
            ref_cube = self._cutout_background_cube(dataset)
        except NoDataAvailableError:
            ref_cube = self.reference

        exposure_cube = reproject_exposure(dataset.exposure, ref_cube)

        exposure_weighted = exposure_cube.empty_like(exposure_cube)
        energies = exposure_weighted.energies('center')

        weights = self.spectral_model(energies)
        exposure_weighted.data = exposure_cube.data * weights.reshape(-1, 1, 1)

        exposure = exposure_weighted.sky_image_integral(emin=p['emin'], emax=p['emax'])
        # TODO: check why fixing the unit is needed
        exposure.data = exposure.data.to('cm2 s').value
        exposure.unit = u.Unit('cm2 s')
        exposure.name = 'exposure'
        return exposure.reproject(self.reference)

    def _psf_image(self, dataset, binsz=0.02):
        """
        Compute fermi PSF image.
        """
        p = self.parameters
        npix = p['rad_max'].to('deg').value / binsz
        npix = np.round(npix).astype('int')
        psf_image = SkyImage.empty(nxpix=npix, nypix=npix, binsz=binsz)

        psf = dataset.psf
        erange = u.Quantity((p['emin'], p['emax']))
        psf_mean = psf.table_psf_in_energy_band(erange, spectrum=self.spectral_model)

        coordinates = psf_image.coordinates()
        offset = coordinates.separation(psf_image.center)
        psf_image.data = psf_mean.evaluate(offset)

        # normalize PSF
        psf_image.data = (psf_image.data / psf_image.data.sum()).value
        return psf_image

    def run(self, dataset, which='all'):
        """Estimate sky images.

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
        images : `~gammapy.image.SkyImageList`
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


def reproject_exposure(exposure, ref_cube):
    """Helper function to reproject exposure to a reference cube.

    TODO: this is a temp solution, as long as we use HpxNDMap objects for exposure
    and SkyCube objects otherwise. This should be changed to use WcsNDMap
    instead of SkyCube in the future.

    Parameters
    ----------
    exposure : `~gammapy.maps.HpxNDMap`
        Exposure cube from gtexmpcube2
    ref_cube : `~gammapy.cube.SkyCube`
        Reference cube to reproject to

    Returns
    -------
    exposure : `~gammapy.cube.SkyCube`
        Exposure reprojected onto reference cube.
    """
    from ..cube import SkyCube
    exposure_cube = SkyCube.empty_like(ref_cube)

    ref_image = ref_cube.sky_image_ref
    geom = ref_image.to_wcs_nd_map().geom
    coords = geom.get_coord()
    for idx, energy in enumerate(ref_cube.energies().value):
        coords_hpx = coords[0], coords[1], energy
        vals = exposure.interp_by_coord(coords_hpx, interp='linear')
        exposure_cube.data[idx] = vals.reshape(ref_image.data.shape)

    exposure_cube.data = exposure_cube.data * u.Unit('cm2 s')

    return exposure_cube

