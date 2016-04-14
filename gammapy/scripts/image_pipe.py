# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import numpy as np
from scipy import interpolate
from astropy.units import Quantity
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import SkyCoord, Angle
from ..utils.energy import Energy, EnergyBounds
from ..background import fill_acceptance_image
from ..image import SkyMap, SkyMapCollection, disk_correlate
from ..stats import significance

__all__ = ['ImageAnalysis']

log = logging.getLogger(__name__)


class ImageAnalysis(object):
    """Gammapy 2D image based analysis.

    Parameters
    ----------
    empty_image : `~gammapy.image.SkyMap`
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
    """

    def __init__(self, empty_image=None,
                 energy_band=None, offset_band=None,
                 data_store=None, obs_table = None, exclusion_mask=None):
        self.maps = SkyMapCollection()

        self.data_store = data_store

        self.obs_table = obs_table
        self.energy_band = energy_band
        self.offset_band = offset_band

        self.empty_image = empty_image
        self.header = self.empty_image.to_image_hdu().header
        if exclusion_mask:
            self.maps['exclusion'] = exclusion_mask
        self.pass_run = []

    def make_counts(self, obs_id):
        """Fill the counts image for the events of one observation.

        Parameters
        ----------
        obs_id : int
            Observation ID

        """
        log.debug('Making counts image ...')
        counts_map = SkyMap.empty_like(self.empty_image)
        obs = self.data_store.obs(obs_id=obs_id)
        events = obs.events.select_energy(self.energy_band)
        events = events.select_offset(self.offset_band)
        if(len(events)!=0):
            counts_map.fill(events=events)
            counts_map.data = counts_map.data.value
            self.maps["counts"] = counts_map
        else:
            print(obs_id)
            self.pass_run.append(obs_id)

    def make_total_counts(self):
        """Stack the total count from the observation in the 'DataStore'

        """
        total_counts = SkyMap.empty_like(self.empty_image)
        for obs_id in self.obs_table['OBS_ID']:
            self.make_counts(obs_id)
            total_counts.data += self.maps["counts"].data
        self.maps["total_counts"] = total_counts

    def make_background(self, obs_id, bkg_norm=True):
        """Make the background map for one observation from a bkg model

        Parameters
        ----------
        obs_id : int
            Observation ID
        bkg_norm : bool
            If true, apply the scaling factor  to the bkg map


        """
        log.debug('Making background image ...')
        bkg_map = SkyMap.empty_like(self.empty_image)
        obs = self.data_store.obs(obs_id=obs_id)
        table = obs.bkg.acceptance_curve_in_energy_band(energy_band=self.energy_band)
        center = obs.pointing_radec.galactic
        bkg_hdu = fill_acceptance_image(self.header, center, table["offset"], table["Acceptance"], self.offset_band[1])
        livetime = obs.observation_live_time_duration
        bkg_map.data = Quantity(bkg_hdu.data, table["Acceptance"].unit) * bkg_map.solid_angle() * livetime
        bkg_map.data = bkg_map.data.decompose()
        bkg_map.data = bkg_map.data.value
        self.maps["bkg"] = bkg_map
        if bkg_norm:
            self.make_counts(obs_id)
            scale = self.background_norm_factor(obs_id, self.maps["counts"], bkg_map)
            self.maps["bkg"].data = scale * self.maps["bkg"].data

    def background_norm_factor(self, obs_id, counts, bkg):
        """Determine the scaling factor to apply to the background map by comparing the events in the counts maps
        and the bkg map outside the exclusion maps

        Parameters
        ----------
        obs_id : int
            Observation ID
        counts : `~gammapy.image.SkyMap`
            counts image
        bkg : `~gammapy.image.SkyMap`
            bkg image

        Returns
        -------
        scale : float
            scaling factor between the counts and the bkg maps outside the exclusion region
        """
        counts_sum = np.sum(counts.data * self.maps['exclusion'].data)
        bkg_sum = np.sum(bkg.data * self.maps['exclusion'].data)
        scale = counts_sum / bkg_sum
        return scale

    def make_total_bkg(self, bkg_norm=True):
        """Stack the total bkg from the observation in the 'DataStore'.

        Parameters
        ----------
        bkg_norm : bool
            If true, apply the scaling factor  to the bkg map
        """
        total_bkg = SkyMap.empty_like(self.empty_image)
        for obs_id in self.obs_table['OBS_ID']:
            if obs_id in self.pass_run:
                    continue
            self.make_background(obs_id, bkg_norm)
            total_bkg.data += self.maps["bkg"].data
        self.maps["total_bkg"] = total_bkg

    def make_significance(self, radius):
        """Make the significance image from the counts and bkg images.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.

        Returns
        -------
        s_maps : `~gammapy.image.SkyMap`
            significance map

        """
        log.debug('Making significance image ...')
        s_maps = SkyMap.empty_like(self.empty_image)
        counts = disk_correlate(self.maps["total_counts"].data, radius)
        bkg = disk_correlate(self.maps["total_bkg"].data, radius)
        s = significance(counts, bkg)
        s_maps.data = s
        self.maps["significance"] = s_maps
        return s_maps

    def make_maps(self, radius, bkg_norm=True, spectral_index = 2.3):
        """Compute the counts, bkg, exlusion_mask and significance images for a set of observation

        Parameters
        ----------
        radius : float
            Disk radius in pixels for the significance map.
        bkg_norm : bool
            If true, apply the scaling factor to the bkg map
        spectral_index : int
            spectral index of the source spectrum
        """
        self.make_total_counts()
        self.make_total_bkg(bkg_norm)
        self.make_significance(radius)
        self.make_total_excess()
        self.make_total_exposure()

    def make_psf(self):
        log.info('Making PSF ...')

    def make_total_excess(self):
        """Compute excess between counts and bkg map."""
        self.maps["total_excess"] = SkyMap.empty_like(self.empty_image)
        excess = self.maps["total_counts"].data - self.maps["total_bkg"].data
        self.maps["total_excess"].data = excess

    def make_exposure_one_obs(self, obs_id, spectral_index = 2.3):
        """Compute the exposure map for one observation.

        Parameters
        ----------
        obs_id : int
            Number of the observation
        spectral_index : int

        """


        exposure = SkyMap.empty_like(self.empty_image)
        xpix_coord_grid, ypix_coord_grid = exposure.coordinates(coord_type='pix')
        # calculate pixel offset from center (in world coordinates)
        coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, exposure.wcs, origin=0)

        center = SkyCoord.from_pixel(self.header["NAXIS1"]/2., self.header["NAXIS2"]/2., exposure.wcs)
        offset = coord.separation(center)

        obs = self.data_store.obs(obs_id=obs_id)
        livetime = obs.observation_live_time_duration

        energy = EnergyBounds(np.linspace(self.energy_band[0].value, self.energy_band[1].value,100), self.energy_band.unit)
        energy_band = energy.bands
        energy_bin = energy.lin_centers
        eref = EnergyBounds(self.energy_band).lin_centers
        spectrum = (energy_bin/eref)**(-spectral_index)
        aeff2d = obs.aeff
        offset_tab = Angle(np.linspace(self.offset_band[0].value,self.offset_band[1].value,10),self.offset_band.unit)
        exposure_tab = np.sum(aeff2d.evaluate(offset_tab, energy_bin).to("cm2")*spectrum*energy_band,axis=1)
        norm = np.sum(spectrum*energy_band)
        exposure_tab /= norm
        f = interpolate.interp1d(offset_tab, exposure_tab, bounds_error = False, fill_value = 0)
        exposure.data= f(offset)
        exposure.data *= livetime
        exposure.data[offset >= self.offset_band[1]] = 0
        self.maps["exposure"] = exposure

    def make_exposure_one_obs2(self, obs_id, spectral_index = 2.3):
        """Compute the exposure map for one observation.

        Parameters
        ----------
        obs_id : int
            Number of the observation
        spectral_index : int

        """


        exposure = SkyMap.empty_like(self.empty_image)
        xpix_coord_grid, ypix_coord_grid = exposure.coordinates(coord_type='pix')
        # calculate pixel offset from center (in world coordinates)
        coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, exposure.wcs, origin=0)

        center = SkyCoord.from_pixel(self.header["NAXIS1"]/2., self.header["NAXIS2"]/2., exposure.wcs)
        offset = coord.separation(center)
        obs = self.data_store.obs(obs_id=obs_id)
        livetime = obs.observation_live_time_duration

        energy = EnergyBounds(np.linspace(self.energy_band[0].value, self.energy_band[1].value,100),
                              self.energy_band.unit)
        energy_band = energy.bands
        energy_bin = energy.lin_centers
        eref = EnergyBounds(self.energy_band).lin_centers

        spectrum = (energy_bin/eref)**(-spectral_index)
        aeff2d = obs.aeff
        offset_tab = Angle(np.linspace(self.offset_band[0].value,self.offset_band[1].value,10),self.offset_band.unit)
        exposure_tab = np.sum(aeff2d.evaluate(offset_tab, energy_bin).to("cm2")*spectrum*energy_band,axis=1)
        f = interpolate.interp1d(offset_tab, exposure_tab, bounds_error = False, fill_value = 0)
        exposure.data= f(offset)
        exposure.data *= livetime
        exposure.data[offset >= self.offset_band[1]] = 0
        self.maps["exposure2"] = exposure

    def make_total_exposure(self, spectral_index = 2.3):
        """Compute the exposure map for all the observations in the obs_table.

        Parameters
        ----------
        spectral_index : int
            spectral index of the source spectrum

        """
        exposure_map = SkyMap.empty_like(self.empty_image)
        exposure_map2 = SkyMap.empty_like(self.empty_image)
        for obs_id in self.obs_table['OBS_ID']:
                if obs_id in self.pass_run:
                    continue
                self.make_exposure_one_obs(obs_id, spectral_index)
                self.make_exposure_one_obs2(obs_id, spectral_index)
                exposure_map.data += self.maps["exposure"].data
                exposure_map2.data += self.maps["exposure2"].data
        self.maps["total_exposure"] = exposure_map
        self.maps["total_exposure2"] = exposure_map2
