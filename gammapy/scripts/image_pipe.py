# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.io import fits
from ..utils.scripts import get_parser
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
    energy_band : `~gammapy.utils.energy.Energy
        Energy band for which we want to compute the image
    offset_band :`astropy.coordinates.Angle`
        Offset Band where you compute the image
    data_store : `~gammapy.data.DataStore`
        `DataStore` where are situated the events
    counts_image : ~gammapy.image.SkyMap`
            count image
    bkg_image : ~gammapy.image.SkyMap`
            bkg image
    """

    def __init__(self, empty_image=None,
                 energy_band=None, offset_band=None,
                 data_store=None, counts_image=None, bkg_image=None):
        self.maps = SkyMapCollection()
        if counts_image:
            self.maps["total_counts"] = counts_image
        else:
            self.data_store = data_store
            self.energy_band = energy_band
            self.offset_band = offset_band

            self.empty_image = empty_image
            self.header = self.empty_image.to_image_hdu().header
        if bkg_image:
            self.maps["total_bkg"] = bkg_image

    def make_counts(self, obs_id):
        """Fill the counts image for the events of one observation.

        Parameters
        ----------
        obs_id : int
            Number of the observation

        """
        self.maps["counts"] = SkyMap.empty_like(self.empty_image)
        obs = self.data_store.obs(obs_id=obs_id)
        events = obs.events.select_energy(self.energy_band)
        events = events.select_offset(self.offset_band)
        self.maps["counts"].fill(events=events)
        self.maps["counts"].data = self.maps["counts"].data.value
        log.info('Making counts image ...')

    def make_total_counts(self):
        """Stack the total count from the observation in the 'DataStore'

        """
        self.maps["total_counts"] = SkyMap.empty_like(self.empty_image)
        for obs_id in self.data_store.obs_table['OBS_ID']:
            self.make_counts(obs_id)
            self.maps["total_counts"].data += self.maps["counts"].data

    def make_background(self, obs_id):
        """Make the background map for one observation from a bkg model

        Parameters
        ----------
        obs_id : int
            Number of the observation

        """
        self.maps["bkg"] = SkyMap.empty_like(self.empty_image)
        obs = self.data_store.obs(obs_id=obs_id)
        table = obs.bkg.acceptance_curve_in_energy_band(energy_band=self.energy_band)
        center = obs.pointing_radec.galactic
        bkg_hdu = fill_acceptance_image(self.header, center, table["offset"], table["Acceptance"], self.offset_band[1])
        livetime = obs.observation_live_time_duration
        self.maps["bkg"].data = Quantity(bkg_hdu.data, table["Acceptance"].unit) * self.maps[
            "bkg"].solid_angle() * livetime
        self.maps["bkg"].data = self.maps["bkg"].data.decompose()
        self.maps["bkg"].data = self.maps["bkg"].data.value

        log.info('Making background image ...')

    def make_background_normalized_offcounts(self, obs_id, exclusion_mask):
        """Normalized the background compare to te events in the counts maps outside the exclusion maps

        Parameters
        ----------
        exclusion_mask : `~gammapy.image.ExclusionMask`
            Exclusion regions


        """
        self.make_counts(obs_id)
        self.make_background(obs_id)
        counts_sum = np.sum(self.maps["counts"].data * exclusion_mask.data)
        bkg_sum = np.sum(self.maps["bkg"].data * exclusion_mask.data)
        scale = counts_sum / bkg_sum
        self.maps["bkg"].data = scale * self.maps["bkg"].data
        self.maps['exclusion'] = exclusion_mask

    def make_total_bkg(self, exclusion_mask):
        """Stack the total bkg from the observation in the 'DataStore'

        Parameters
        ----------
        exclusion_mask : `~gammapy.image.ExclusionMask`
            Exclusion regions

        """
        self.maps["total_bkg"] = SkyMap.empty_like(self.empty_image)
        for obs_id in self.data_store.obs_table['OBS_ID']:
            self.make_background_normalized_offcounts(obs_id, exclusion_mask)
            self.maps["total_bkg"].data += self.maps["bkg"].data

    def make_significance(self, radius, counts_image=None, bkg_image=None):
        """Make the dignificance image from the counts and kbg images.

        Parameters
        ----------
        radius : float
            Disk radius in pixels.
        counts_image : `~gammapy.image.SkyMap`
            count image
        bkg_image : `~gammapy.image.SkyMap`
            bkg image

        """

        if not counts_image:
            self.maps["significance"] = SkyMap.empty_like(self.empty_image)
            counts_image = self.maps["total_counts"]
        else:
            self.maps["significance"] = SkyMap.empty_like(counts_image)
        if not bkg_image:
            bkg_image = self.maps["total_bkg"]

        counts = disk_correlate(counts_image.data, 10)
        bkg = disk_correlate(bkg_image.data, 10)
        s = significance(counts, bkg)
        self.maps["significance"].data = s
        log.info('Making significance image ...')

    def make_maps(self, exclusion_mask, radius):
        """Compute the counts, bkg, exlusion_mask and significance images for a set of observation

        Parameters
        ----------
        exclusion_mask : `~gammapy.image.ExclusionMask`
            Exclusion regions
        radius : float
            Disk radius in pixels for the significance map.

        """
        self.make_total_counts()
        self.make_total_bkg(exclusion_mask)
        self.make_significance(radius)

    def make_psf(self):
        log.info('Making PSF ...')

    def make_exposure(self):
        log.info('Making Exposure ...')
