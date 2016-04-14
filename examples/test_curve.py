"""Example how to make an acceptance curve and background model image.
"""
import numpy as np
from astropy.table import Table
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity
from gammapy.datasets import gammapy_extra
from gammapy.background import EnergyOffsetBackgroundModel
from gammapy.utils.energy import EnergyBounds, Energy
from gammapy.data import DataStore
from gammapy.utils.axis import sqrt_space
from gammapy.image import bin_events_in_image, disk_correlate, SkyMap, ExclusionMask, SkyMapCollection
from gammapy.background import fill_acceptance_image
from gammapy.region import SkyCircleRegion
from gammapy.stats import significance
from gammapy.background import OffDataBackgroundMaker
from gammapy.scripts import ImageAnalysis
from gammapy.data import ObservationTable
# from gammapy.detect import compute_ts_map
import matplotlib.pyplot as plt
from gammapy.utils.scripts import make_path

plt.ion()

import logging

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


def make_excluded_sources():
    # centers = SkyCoord([84, 82], [23, 21], unit='deg')
    centers = SkyCoord([83.63, 83.63], [22.01, 22.01], unit='deg', frame='icrs')
    radius = Angle('0.3 deg')
    sources = SkyCircleRegion(pos=centers, radius=radius)
    catalog = Table()
    catalog["RA"] = sources.pos.data.lon
    catalog["DEC"] = sources.pos.data.lat
    catalog["Radius"] = sources.radius
    return catalog


def make_model():
    dir = str(gammapy_extra.dir) + '/datasets/hess-crab4-hd-hap-prod2'
    data_store = DataStore.from_dir(dir)
    obs_table = data_store.obs_table
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = sqrt_space(start=0, stop=2.5, num=100) * u.deg

    excluded_sources = make_excluded_sources()

    multi_array = EnergyOffsetBackgroundModel(ebounds, offset)
    multi_array.fill_obs(obs_table, data_store, excluded_sources)
    # multi_array.fill_obs(obs_table, data_store)
    multi_array.compute_rate()
    bgarray = multi_array.bg_rate
    energy_range = Energy([1, 10], 'TeV')
    table = bgarray.acceptance_curve_in_energy_band(energy_range, energy_bins=10)

    multi_array.write('energy_offset_array.fits', overwrite=True)
    table.write('acceptance_curve.fits', overwrite=True)


def make_image():
    table = Table.read('acceptance_curve.fits')
    table.pprint()
    center = SkyCoord(83.63, 22.01, unit='deg').galactic

    counts_image = SkyMap.empty(nxpix=1000, nypix=1000, binsz=0.01, xref=center.l.deg, yref=center.b.deg,
                                proj='TAN').to_image_hdu()
    bkg_image = counts_image.copy()
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')

    for events in data_store.load_all("events"):
        center = events.pointing_radec.galactic
        livetime = events.observation_live_time_duration
        solid_angle = Angle(0.01, "deg") ** 2

        counts_image.data += bin_events_in_image(events, counts_image).data

        # interp_param = dict(bounds_error=False, fill_value=None)

        acc_hdu = fill_acceptance_image(bkg_image.header, center, table["offset"], table["Acceptance"])
        acc = Quantity(acc_hdu.data, table["Acceptance"].unit) * solid_angle * livetime
        bkg_image.data += acc.decompose()
        print(acc.decompose().sum())

    counts_image.writeto("counts_image_save.fits", clobber=True)
    bkg_image.writeto("bkg_image_save.fits", clobber=True)

    # result = compute_ts_map(counts_stacked_image.data, bkg_stacked_image.data,
    #  maps['ExpGammaMap'].data, kernel)


def make_significance_image():
    counts_image = fits.open("counts_image.fits")[1]
    bkg_image = fits.open("bkg_image.fits")[1]
    counts = disk_correlate(counts_image.data, 10)
    bkg = disk_correlate(bkg_image.data, 10)
    s = significance(counts, bkg)
    s_image = fits.ImageHDU(data=s, header=counts_image.header)
    s_image.writeto("significance_image.fits", clobber=True)


def plot_model():
    multi_array = EnergyOffsetBackgroundModel.read('energy_offset_array.fits')
    table = Table.read('acceptance_curve.fits')
    plt.figure(1)
    multi_array.counts.plot_image()
    plt.figure(2)
    multi_array.livetime.plot_image()
    plt.figure(3)
    multi_array.bg_rate.plot_image()
    plt.figure(4)
    plt.plot(table["offset"], table["Acceptance"])
    plt.xlabel("offset (deg)")
    plt.ylabel("Acceptance (s-1 sr-1)")

    input()


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
    image = SkyMap.empty(nxpix=250, nypix=250, binsz=0.02, xref=center.l.deg,
                         yref=center.b.deg, proj='TAN', coordsys='GAL')


    refheader = image.to_image_hdu().header
    exclusion_mask = ExclusionMask.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    exclusion_mask = exclusion_mask.reproject(refheader=refheader)
    images = ImageAnalysis(image, energy_band=energy_band, offset_band=offset_band, data_store=data_store,
                           obs_table=data_store.obs_table, exclusion_mask=exclusion_mask)

    images.make_maps(radius=10.)
    #images.make_total_exposure()
    filename = 'fov_bg_maps.fits'
    log.info('Writing {}'.format(filename))

    images.maps.write(filename, clobber=True)


if __name__ == '__main__':
    # make_model()
    # plot_model()
    # make_image()
    # make_significance_image()

    # make_bg_model_two_groups()
    make_image_from_2d_bg()
