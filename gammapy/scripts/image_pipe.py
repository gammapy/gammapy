# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import numpy as np
from astropy.units import Quantity
from astropy.table import QTable, Column
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import Angle
from ..utils.energy import EnergyBounds
from ..background import fill_acceptance_image
from ..image import SkyImage, SkyImageCollection, disk_correlate
from ..stats import significance

__all__ = ['ObsImage',
           'MosaicImage']

log = logging.getLogger(__name__)


class ObsImage(object):
    """Gammapy 2D image based analysis for one observation.

    The computed images are stored in a ``maps`` attribute of type `~gammapy.image.SkyImageCollection`
    with the following keys:

    * counts : counts for one obs
    * bkg : bkg model for one obs
    * exposure : exposure for one obs
    * excess : excess for one obs
    * significance : significance for one obs

    Parameters
    ----------
    obs : `~gammapy.data.DataStoreObservation`
        `DataStoreObservation` for the observation
    empty_image : `~gammapy.image.SkyImage`
            ref to an empty image
    energy_band : `~gammapy.utils.energy.Energy`
        Energy band for which we want to compute the image
    offset_band : `astropy.coordinates.Angle`
        Offset Band where you compute the image
    exclusion_mask : `~gammapy.image.ExclusionMask`
            Exclusion regions
    ncounts_min : int
            Minimum counts required for the observation
    """

    def __init__(self, obs, empty_image,
                 energy_band, offset_band, exclusion_mask=None, ncounts_min=0):
        # Select the events in the given energy and offset range
        self.energy_band = energy_band
        self.offset_band = offset_band
        events = obs.events
        self.obs_id = events.meta["OBS_ID"]
        events = events.select_energy(self.energy_band)
        self.events = events.select_offset(self.offset_band)

        self.maps = SkyImageCollection()
        self.empty_image = empty_image
        self.header = self.empty_image.to_image_hdu().header
        if exclusion_mask:
            self.maps['exclusion'] = exclusion_mask

        self.ncounts_min = ncounts_min
        self.aeff = obs.aeff
        self.edisp = obs.edisp
        self.psf = obs.psf
        self.bkg = obs.bkg
        self.obs_center = obs.pointing_radec
        self.livetime = obs.observation_live_time_duration

    def counts_map(self):
        """Fill the counts image for the events of one observation."""
        counts_map = SkyImage.empty_like(self.empty_image)
        if len(self.events) > self.ncounts_min:
            counts_map.fill(value=self.events)
        else:
            log.warn('Too few counts, there is only {} events and you requested a minimal counts number of {}'.
                     format(len(self.events), self.ncounts_min))
        self.maps["counts"] = counts_map

    def bkg_map(self, bkg_norm=True):
        """Make the background map for one observation from a bkg model.

        Parameters
        ----------
        bkg_norm : bool
            If true, apply the scaling factor from the number of counts outside the exclusion region to the bkg map
        """
        bkg_map = SkyImage.empty_like(self.empty_image)
        table = self.bkg.acceptance_curve_in_energy_band(energy_band=self.energy_band)
        center = self.obs_center.galactic
        bkg_hdu = fill_acceptance_image(self.header, center, table["offset"], table["Acceptance"], self.offset_band[1])
        sep_pix = Angle(np.abs(self.header["CDELT2"]), "deg")
        solid_angle = sep_pix ** 2
        # TODO: used the solid angle method when it will be fixed in SkyImage
        # bkg_map.data = Quantity(bkg_hdu.data, table["Acceptance"].unit) * bkg_map.solid_angle() * self.livetime
        bkg_map.data = Quantity(bkg_hdu.data, table["Acceptance"].unit) * solid_angle * self.livetime
        bkg_map.data = bkg_map.data.decompose()
        bkg_map.data = bkg_map.data.value
        if bkg_norm:
            scale = self.background_norm_factor(self.maps["counts"], bkg_map)
            bkg_map.data = scale * bkg_map.data

        self.maps["bkg"] = bkg_map

    def make_1d_expected_counts(self, spectral_index=2.3, for_integral_flux=False):
        """Compute the 1D exposure table for one observation for an offset table

        Parameters
        ----------
        spectral_index : float
            Assumed power-law spectral index
        for_integral_flux : bool
            True if you want that the total excess / exposure gives the integrated flux

        Returns
        -------
        table : `astropy.table.QTable`
            Two columns: offset in the FOV "theta" and expected counts "npred"
        """
        energy = EnergyBounds.equal_log_spacing(self.energy_band[0].value, self.energy_band[1].value, 100,
                                                self.energy_band.unit)
        energy_band = energy.bands
        energy_bin = energy.log_centers
        eref = EnergyBounds(self.energy_band).log_centers
        spectrum = (energy_bin / eref) ** (-spectral_index)
        offset = Angle(np.linspace(self.offset_band[0].value, self.offset_band[1].value, 10), self.offset_band.unit)
        arf = self.aeff.evaluate(offset=offset, energy=energy_bin).T
        npred = np.sum(arf * spectrum * energy_band, axis=1)
        npred *= self.livetime

        if for_integral_flux:
            norm = np.sum(spectrum * energy_band)
            npred /= norm

        table = QTable()
        table['theta'] = offset
        table['npred'] = npred

        return table

    def exposure_map(self, spectral_index=2.3, for_integral_flux=False):
        r"""Compute the exposure map for one observation.

        Excess/exposure will give the differential flux at the energy Eref at the middle of the ``self.energy_band``

        If ``for_integral_flux`` is true, it will give the integrated flux over the ``self.energy_band``

        Exposure is define as follow:

        .. math ::

            EXPOSURE = \int_{E_1}^{E_2} A(E) \phi(E) * T \, dE

        with ``T`` the observation livetime, ``A(E)`` the effective area,
        the energy integration range :math:`[E_1,E_2]` given by ``self.energy_range``
        and assuming a power law for the flux :math:`\phi(E) = \phi_{Eref} \times \frac{E}{E_{ref}}^{\gamma}`
        with :math:`\gamma` the spectral index of the assumed power law.

        If ``for_integral_flux`` is true,
        :math:`EXPOSURE = \int_{E_1}^{E_2} A(E) \phi_{E} * T \, dE / \int \phi_{E} \, dE`

        Parameters
        ----------
        spectral_index : float
            Assumed power-law spectral index
        for_integral_flux : bool
            True if you want that the total excess / exposure gives the integrated flux
        """
        from scipy.interpolate import interp1d
        # TODO: should be re-implemented using the exposure_cube function
        table = self.make_1d_expected_counts(spectral_index, for_integral_flux)
        exposure = SkyImage.empty_like(self.empty_image, unit=table["npred"].unit)

        # Determine offset value for each pixel of the map
        xpix_coord_grid, ypix_coord_grid = exposure.coordinates_pix()
        # calculate pixel offset from center (in world coordinates)
        coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, exposure.wcs, origin=0)
        offset = coord.separation(self.obs_center)

        # Interpolate for the offset of each pixel
        f = interp1d(table["theta"], table["npred"], bounds_error=False, fill_value=0)
        exposure.data = f(offset)
        exposure.data[offset >= self.offset_band[1]] = 0
        self.maps["exposure"] = exposure

    def background_norm_factor(self, counts, bkg):
        """Determine the scaling factor to apply to the background map.

        Compares the events in the counts maps and the bkg map outside the exclusion maps.

        Parameters
        ----------
        counts : `~gammapy.image.SkyImage`
            counts image
        bkg : `~gammapy.image.SkyImage`
            bkg image

        Returns
        -------
        scale : float
            scaling factor between the counts and the bkg maps outside the exclusion region.
        """
        counts_sum = np.sum(counts.data * self.maps['exclusion'].data)
        bkg_sum = np.sum(bkg.data * self.maps['exclusion'].data)
        scale = counts_sum / bkg_sum

        return scale

    def significance_image(self, radius):
        """Make the significance image from the counts and bkg images.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        """
        s_map = SkyImage.empty_like(self.empty_image)
        counts = disk_correlate(self.maps["counts"].data, radius)
        bkg = disk_correlate(self.maps["bkg"].data, radius)
        s = significance(counts, bkg)
        s_map.data = s

        self.maps["significance"] = s_map

    def excess_image(self):
        """Compute excess between counts and bkg map."""
        total_excess = SkyImage.empty_like(self.empty_image)
        total_excess.data = self.maps["counts"].data - self.maps["bkg"].data
        self.maps["excess"] = total_excess


class MosaicImage(object):
    """Gammapy 2D image based analysis for a set of observations.

    The computed images are stored in a ``maps`` attribute of type `~gammapy.image.SkyImageCollection`
    with the following keys:

    * counts : counts for the set of obs
    * bkg : bkg model for the set of obs
    * exposure : exposure for the set of obs
    * excess : excess for the set of obs
    * significance : significance for the set of obs

    Parameters
    ----------
    empty_image : `~gammapy.image.SkyImage`
            ref to an empty image
    energy_band : `~gammapy.utils.energy.Energy`
        Energy band for which we want to compute the image
    offset_band : `astropy.coordinates.Angle`
        Offset Band where you compute the image
    data_store : `~gammapy.data.DataStore`
        `DataStore` where are situated the events
    obs_table : `~astropy.table.Table`
            Required columns: OBS_ID
    exclusion_mask : `~gammapy.image.ExclusionMask`
            Exclusion regions
    ncounts_min : int
            Minimum counts required for the observation
    """

    def __init__(self, empty_image=None,
                 energy_band=None, offset_band=None,
                 data_store=None, obs_table=None, exclusion_mask=None, ncounts_min=0):

        self.maps = SkyImageCollection()

        self.data_store = data_store
        self.obs_table = obs_table
        self.energy_band = energy_band
        self.offset_band = offset_band

        self.empty_image = empty_image
        self.header = self.empty_image.to_image_hdu().header
        self.exclusion_mask = exclusion_mask
        if exclusion_mask:
            self.maps['exclusion'] = exclusion_mask
        self.ncounts_min = ncounts_min
        self.psfmeantab = None
        self.thetapsf = None

    def make_images(self, make_background_image=False, bkg_norm=True, spectral_index=2.3, for_integral_flux=False,
                    radius=10):
        """Compute the counts, bkg, exposure, excess and significance images for a set of observation.

        Parameters
        ----------
        make_background_image : bool
            True if you want to compute the background and exposure maps
        bkg_norm : bool
            If true, apply the scaling factor to the bkg map
        spectral_index : float
            Assumed power-law spectral index
        for_integral_flux : bool
            True if you want that the total excess / exposure gives the integrated flux
        radius : float
            Disk radius in pixels for the significance map.
        """

        total_counts = SkyImage.empty_like(self.empty_image)
        if make_background_image:
            total_bkg = SkyImage.empty_like(self.empty_image)
            total_exposure = SkyImage.empty_like(self.empty_image)
        for obs_id in self.obs_table['OBS_ID']:
            obs = self.data_store.obs(obs_id)
            obs_image = ObsImage(obs, self.empty_image, self.energy_band, self.offset_band,
                                 self.exclusion_mask, self.ncounts_min)
            if len(obs_image.events) <= self.ncounts_min:
                continue
            else:
                obs_image.counts_map()
                total_counts.data += obs_image.maps["counts"].data
                if make_background_image:
                    obs_image.bkg_map(bkg_norm)
                    obs_image.exposure_map(spectral_index, for_integral_flux)
                    total_bkg.data += obs_image.maps["bkg"].data
                    total_exposure.data += obs_image.maps["exposure"].data
        self.maps["counts"] = total_counts
        if make_background_image:
            self.maps["bkg"] = total_bkg
            self.maps["exposure"] = total_exposure
            self.significance_image(radius)
            self.excess_image()

    def significance_image(self, radius):
        """Make the significance image from the counts and bkg images.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        """
        s_map = SkyImage.empty_like(self.empty_image)
        counts = disk_correlate(self.maps["counts"].data, radius)
        bkg = disk_correlate(self.maps["bkg"].data, radius)
        s = significance(counts, bkg)
        s_map.data = s

        self.maps["significance"] = s_map

    def excess_image(self):
        """Compute excess between counts and bkg map."""
        total_excess = SkyImage.empty_like(self.empty_image)
        total_excess.data = self.maps["counts"].data - self.maps["bkg"].data
        self.maps["excess"] = total_excess
