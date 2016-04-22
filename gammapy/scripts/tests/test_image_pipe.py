"""Example how to make an acceptance curve and background model image.
"""
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
from gammapy.image import bin_events_in_image, disk_correlate, SkyMap, ExclusionMask
from gammapy.background import fill_acceptance_image
from gammapy.region import SkyCircleRegion
from gammapy.stats import significance
from gammapy.background import OffDataBackgroundMaker
from gammapy.scripts import ImageAnalysis




def test_image_pipe(tmpdir):
    tmpdir=str(tmpdir)
    from subprocess import call
    outdir = tmpdir
    outdir2 = outdir + '/background'

    cmd = 'mkdir -p {}'.format(outdir2)
    print('Executing: {}'.format(cmd))
    call(cmd, shell=True)

    cmd = 'cp -r $GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/ {}'.format(outdir)
    print('Executing: {}'.format(cmd))
    call(cmd, shell=True)

    data_store = DataStore.from_dir(tmpdir)

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
    
    tmpdir=str(tmpdir)
    center = SkyCoord(83.63, 22.01, unit='deg').galactic
    energy_band = Energy([1, 10], 'TeV')
    offset_band = Angle([0, 2.49], 'deg')
    data_store = DataStore.from_dir(tmpdir)

    # TODO: fix `binarize` implementation
    # exclusion_mask = exclusion_mask.binarize()
    image = SkyMap.empty(nxpix=250, nypix=250, binsz=0.02, xref=center.l.deg,
                         yref=center.b.deg, proj='TAN', coordsys='GAL')

    refheader = image.to_image_hdu().header
    exclusion_mask = ExclusionMask.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    images = ImageAnalysis(image, energy_band=energy_band, offset_band=offset_band, data_store=data_store,
                           obs_table=data_store.obs_table, exclusion_mask=exclusion_mask)

    images.make_maps(radius=10., bkg_norm=True, spectral_index=2.3, for_integral_flux = True)

