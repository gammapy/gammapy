# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.coordinates import SkyCoord, Angle
from numpy.testing import assert_allclose
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import Energy
from ...data import DataStore
from ...image import SkyImage
from ...background import OffDataBackgroundMaker
from ..image_pipe import StackedObsImageMaker


@requires_dependency('reproject')
@requires_data('gammapy-extra')
def test_image_pipe(tmpdir):
    """Example how to make an acceptance curve and background model image."""
    tmpdir = str(tmpdir)
    from subprocess import call
    outdir = tmpdir
    outdir2 = outdir + '/background'

    cmd = 'mkdir -p {}'.format(outdir2)
    print('Executing: {}'.format(cmd))
    call(cmd, shell=True)

    ds = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2")
    ds.copy_obs(ds.obs_table, tmpdir)
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

    center = SkyCoord(83.63, 22.01, unit='deg').galactic
    energy_band = Energy([1, 10], 'TeV')
    offset_band = Angle([0, 2.49], 'deg')
    data_store = DataStore.from_dir(tmpdir)

    ref_image = SkyImage.empty(nxpix=250, nypix=250, binsz=0.02, xref=center.l.deg,
                               yref=center.b.deg, proj='TAN', coordsys='GAL')

    exclusion_mask = SkyImage.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    exclusion_mask = exclusion_mask.reproject(reference=ref_image)

    # TODO: fix this:
    # Pb with the load psftable for one of the run that is not implemented yet...
    data_store.hdu_table.remove_row(14)

    image_maker = StackedObsImageMaker(
        empty_image=ref_image, energy_band=energy_band, offset_band=offset_band, data_store=data_store,
        obs_table=data_store.obs_table, exclusion_mask=exclusion_mask,
    )
    image_maker.make_images(make_background_image=True, for_integral_flux=True, radius=10.)
    images = image_maker.images

    assert_allclose(images['counts'].data.sum(), 2334.0, atol=3)
    assert_allclose(images['bkg'].data.sum(), 1987.1513636663785, atol=3)
    assert_allclose(images['exposure'].data.sum(), 54190569251987.68, atol=3)
    assert_allclose(images['significance'].lookup(center), 33.707901541600634, atol=3)
    assert_allclose(images['excess'].data.sum(), 346.8486363336217, atol=3)
    assert_allclose(image_maker.table_bkg_scale[0]["bkg_scale"], 0.7502867380744898, rtol=0.03)
    assert_allclose(image_maker.table_bkg_scale[1]["bkg_scale"], 0.7402389407935327, rtol=0.03)
    assert_allclose(image_maker.table_bkg_scale["N_counts"][0], 525.0583940319832, rtol=0.03)
