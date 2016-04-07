# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from ..utils.scripts import get_parser
from ..background import fill_acceptance_image
from ..image import SkyMap

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
        self.offset_band= offset_band
        if not counts_image:
            self.counts_image = SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=self.center.l.deg,
                                         yref=self.center.b.deg, proj='TAN')
        if not bkg_image:
            self.bkg_image = SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=self.center.l.deg,
                                      yref=self.center.b.deg, proj='TAN')
        self.total_counts_image= SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=self.center.l.deg,
                                         yref=self.center.b.deg, proj='TAN')
        self.total_bkg_image= SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=self.center.l.deg,
                                         yref=self.center.b.deg, proj='TAN')
        self.header = self.counts_image.to_image_hdu().header
        self.solid_angle = Angle(0.01, "deg") ** 2

    @classmethod
    def from_yaml(cls, filename):
        """Read config from YAML file."""
        import yaml
        log.info('Reading {}'.format(filename))
        with open(filename) as fh:
            config = yaml.safe_load(fh)
        return cls(config)

    def run(self):
        """Run analysis chain."""
        log.info('Running analysis ...')
        print(self.config['general']['outdir'])
        print(self.config)

    def make_counts(self, obs_id):
        """Fill the counts image for the events of one observation

        Parameters
        ----------
        obs_id : int
            Number of the observation

        """
        obs = self.data_store.obs(obs_id=obs_id)
        events = obs.events.select_energy(self.energy_band)
        events = events.select_offset(offset_band)
        self.counts_image.fill(events=events)
        log.info('Making counts image ...')

    def make_total_counts(self):
        """

        Returns
        -------

        """
        for obs_id in data_store.obs_table['OBS_ID']:
            self.make_counts(obs_id)

    def make_exposure(self):
        log.info('Making exposure image ...')

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
        """

        Parameters
        ----------
        exclusion_mask : `~ggammapy.image.ExclusionMask`

        Returns
        -------


        """
        self.make_background(obs_id)
        counts_sum = np.sum(self.counts_image.data * exclusion_mask.data)
        bkg_sum = np.sum(self.bkg_image.data * exclusion_mask.data)
        scale = counts_sum / bkg_sum
        self.bkg_image.data = scale * self.bkg_image.data

    def make_images(self, exclusion_mask):
        """

        Parameters
        ----------
        obs_id
        exclusion_mask

        Returns
        -------

        """
        counts_image_total = SkyMap.empty_like(counts_image_total)
        #bkg_image_total =
        for obs_id in data_store.obs_table['OBS_ID']:
            self.make_counts(obs_id)
            self.make_background_normalized(obs_id, exclusion_mask)

    def make_images(self):
        maps = SkyMapCollection()
        maps['counts'] = counts_image_total
        maps['bkg'] = bkg_image_total
        maps['exclusion'] = exclusion_mask
        return mask

    def make_psf(self):
        log.info('Making PSF ...')

    def fit_source(self):
        log.info('Fitting image ...')
