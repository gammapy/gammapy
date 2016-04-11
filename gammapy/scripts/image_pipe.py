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


def image_pipe_main(args=None):
    parser = get_parser(ImageAnalysis)
    parser.add_argument('config_file', type=str,
                        help='Config file in YAML format')
    # TODO: add option to dump the default config file
    args = parser.parse_args(args)
    analysis = ImageAnalysis.from_yaml(args.config_file)
    analysis.run()


class ImageAnalysis(object):
    """Gammapy 2D image based analysis.

    center : `astropy.coordinates.SkyCoord`
        Center of the image
    energy_band : `~gammapy.utils.energy.Energy
        Energy band for which we want to compute the image
    offset_band :`astropy.coordinates.Angle`
        Offset Band where you compute the image
    data_store : `~gammapy.data.DataStore`
        `DataStore` where are situated the events
    """

    def __init__(self, center, energy_band, offset_band, data_store, counts_image=None, bkg_image=None):
        self.data_store = data_store
        self.center = center
        self.energy_band = energy_band
        self.offset_band = offset_band
        if not counts_image:
            self.counts_image = SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=self.center.l.deg,
                                             yref=self.center.b.deg, proj='TAN')
        if not bkg_image:
            self.bkg_image = SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=self.center.l.deg,
                                          yref=self.center.b.deg, proj='TAN')
        self.total_counts_image = SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=self.center.l.deg,
                                               yref=self.center.b.deg, proj='TAN')
        self.total_bkg_image = SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=self.center.l.deg,
                                            yref=self.center.b.deg, proj='TAN')
        self.header = self.counts_image.to_image_hdu().header
        self.solid_angle = Angle(0.01, "deg") ** 2

    def make_counts(self, obs_id):
        """Fill the counts image for the events of one observation

        Parameters
        ----------
        obs_id : int
            Number of the observation

        """
        obs = self.data_store.obs(obs_id=obs_id)
        events = obs.events.select_energy(self.energy_band)
        events = events.select_offset(self.offset_band)
        self.counts_image.fill(events=events)
        log.info('Making counts image ...')

    def make_total_counts(self):
        """Stack the total count from the observation in the 'DataStore'

        Returns
        -------

        """
        for obs_id in self.data_store.obs_table['OBS_ID']:
            self.make_counts(obs_id)
            self.total_counts_image.data += self.counts_image.data

    def make_background(self, obs_id):
        obs = self.data_store.obs(obs_id=obs_id)
        table = obs.bkg.acceptance_curve_in_energy_band(energy_band=self.energy_band)
        center = obs.pointing_radec.galactic
        bkg_hdu = fill_acceptance_image(self.header, center, table["offset"], table["Acceptance"], self.offset_band[1])
        livetime = obs.observation_live_time_duration
        self.bkg_image.data = Quantity(bkg_hdu.data, table["Acceptance"].unit) * self.solid_angle * livetime
        self.bkg_image.data = self.bkg_image.data.decompose()

        log.info('Making background image ...')

    def make_background_normalized(self, obs_id, exclusion_mask):
        """Normalized the background compare to te events in the counts maps outside the exclusion maps

        Parameters
        ----------
        exclusion_mask : `~gammapy.image.ExclusionMask`

        Returns
        -------


        """
        self.make_counts(obs_id)
        self.make_background(obs_id)
        counts_sum = np.sum(self.counts_image.data * exclusion_mask.data)
        bkg_sum = np.sum(self.bkg_image.data * exclusion_mask.data)
        scale = counts_sum / bkg_sum
        self.bkg_image.data = scale * self.bkg_image.data

    def make_total_bkg(self, exclusion_mask):
        """Stack the total bkg from the observation in the 'DataStore'

        Parameters
        ----------
        exclusion_mask : `~gammapy.image.ExclusionMask`

        Returns
        -------

        """
        for obs_id in self.data_store.obs_table['OBS_ID']:
            self.make_background_normalized(obs_id, exclusion_mask)
            self.total_bkg_image.data += self.bkg_image.data

    def make_images(self, exclusion_mask):
        """Compute the counts, bkg and exlusion_mask images for a set of observation

        Parameters
        ----------
        exclusion_mask : `~gammapy.image.ExclusionMask`

        Returns
        -------
        maps : `
        """
        self.make_total_counts()
        self.make_total_bkg(exclusion_mask)
        maps = SkyMapCollection()
        maps['counts'] = self.total_counts_image
        maps['bkg'] = self.total_bkg_image
        maps['exclusion'] = exclusion_mask
        return maps

    def make_significance(self, counts_image, bkg_image, radius):
        """

        Parameters
        ----------
        counts_image : ~gammapy.image.SkyMap`
            count image
        bkg_image : ~gammapy.image.SkyMap`
            bkg image
        radius : float
        Disk radius in pixels.

        Returns
        -------

        """
        counts = disk_correlate(counts_image.data, 10)
        bkg = disk_correlate(bkg_image.data, 10)
        s = significance(counts, bkg)
        s_image = fits.ImageHDU(data=s, header=counts_image.to_image_hdu().header)
        return s_image

    def make_psf(self):
        log.info('Making PSF ...')

    def make_exposure(self):
        """
        ny, nx = ref_cube.data.shape[1:]
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
        lon, lat, en = ref_cube.pix2world(xx, yy, 0)
        coord = SkyCoord(lon, lat, frame=ref_cube.wcs.wcs.radesys.lower())  # don't care about energy
        offset = coord.separation(pointing)
        offset = np.clip(offset, Angle(0, 'deg'), offset_max)

        energy = EnergyBounds(ref_cube.energy).log_centers
        exposure = aeff2d.evaluate(offset, energy)
        exposure = np.rollaxis(exposure, 2)
        exposure *= livetime

        expcube = SpectralCube(data=exposure,
                           wcs=ref_cube.wcs,
                           energy=ref_cube.energy)
        return expcube
        """
        wcs = WCS(self.header)
        data = np.zeros((header["NAXIS2"], header["NAXIS1"]))
        image = fits.ImageHDU(data=data, header=header)

        # define grids of pixel coorinates
        xpix_coord_grid, ypix_coord_grid = coordinates(image, world=False)
        # calculate pixel offset from center (in world coordinates)
        coord = pixel_to_skycoord(xpix_coord_grid, ypix_coord_grid, wcs, origin=0)
        pix_off = coord.separation(center)
        log.info('Making exposure image ...')

    def fit_source(self):
        log.info('Fitting image ...')
