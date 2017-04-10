"""Example how to make an acceptance curve and background model image.
"""

from astropy.coordinates import SkyCoord, Angle
from gammapy.utils.energy import Energy
from gammapy.data import DataStore
from gammapy.image import SkyImage
from gammapy.background import OffDataBackgroundMaker
from gammapy.scripts import StackedObsImageMaker
import matplotlib.pyplot as plt

plt.ion()

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def make_bg_model_two_groups():
    from subprocess import call
    outdir = '/Users/jouvin/Desktop/these/temp/bg_model_image/'
    outdir2 = outdir + '/background'

    cmd = 'mkdir -p {}'.format(outdir2)
    print('Executing: {}'.format(cmd))
    call(cmd, shell=True)

    cmd = 'cp -r $GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/ {}'.format(outdir)
    print('Executing: {}'.format(cmd))
    call(cmd, shell=True)

    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')

    bgmaker = OffDataBackgroundMaker(data_store, outdir=outdir2)

    bgmaker.select_observations(selection='all')
    bgmaker.group_observations()
    bgmaker.make_model("2D")
    bgmaker.save_models("2D")

    fn = outdir2 + '/group-def.fits'
    hdu_index_table = bgmaker.make_total_index_table(
        data_store=data_store,
        modeltype='2D',
        out_dir_background_model=outdir2,
        filename_obs_group_table=fn
    )

    fn = outdir + '/hdu-index.fits.gz'
    hdu_index_table.write(fn, overwrite=True)


def make_image_from_2d_bg():
    center = SkyCoord(83.63, 22.01, unit='deg').galactic
    energy_band = Energy([1, 10], 'TeV')
    offset_band = Angle([0, 2.49], 'deg')
    data_store = DataStore.from_dir('/Users/jouvin/Desktop/these/temp/bg_model_image/')

    # TODO: fix `binarize` implementation
    # exclusion_mask = exclusion_mask.binarize()
    image = SkyImage.empty(nxpix=250, nypix=250, binsz=0.02, xref=center.l.deg,
                           yref=center.b.deg, proj='TAN', coordsys='GAL')

    refheader = image.to_image_hdu().header
    exclusion_mask = SkyImage.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    mosaic_images = StackedObsImageMaker(image, energy_band=energy_band, offset_band=offset_band, data_store=data_store,
                                         obs_table=data_store.obs_table, exclusion_mask=exclusion_mask)
    mosaic_images.make_images(make_background_image=True, for_integral_flux=True, radius=10., make_psf = True,
                              region_center=center)
    filename = 'fov_bg_images.fits'
    log.info('Writing {}'.format(filename))
    mosaic_images.images.write(filename, clobber=True)


if __name__ == '__main__':
    # make_bg_model_two_groups()
    make_image_from_2d_bg()
