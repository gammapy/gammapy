# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from ...extern.pathlib import Path
from ...utils.testing import requires_dependency, requires_data
from ...data import DataStore
from ...image import SkyImage
from ...background import OffDataBackgroundMaker
from .. import StackedObsCubeMaker
from .. import SkyCube


def make_empty_cube(emin, emax, enumbins, data_unit=''):
    """Make a reference cube at the Crab nebula position for testing."""
    return SkyCube.empty(
        emin=emin, emax=emax, enumbins=enumbins, eunit='TeV', mode='edges',
        nxpix=250, nypix=250, binsz=0.02,
        xref=184.55974014, yref=-5.78918015,
        proj='TAN', coordsys='GAL', unit=data_unit,
    )


@requires_dependency('reproject')
@requires_data('gammapy-extra')
def test_cube_pipe(tmpdir):
    """Example how to make a Cube analysis from a 2D background model."""
    tmpdir = str(tmpdir)
    outdir = tmpdir
    outdir2 = outdir + '/background'
    Path(outdir2).mkdir()

    ds = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2")
    ds.copy_obs(ds.obs_table, tmpdir)
    data_store = DataStore.from_dir(tmpdir)

    # Create the background model from the 4 Crab observations
    bgmaker = OffDataBackgroundMaker(data_store, outdir=outdir2)

    bgmaker.select_observations(selection='all')
    bgmaker.group_observations()
    bgmaker.make_model("2D")
    bgmaker.save_models("2D")

    fn = outdir2 + '/group-def.fits'

    # New hdu table that contains the link to the background model
    hdu_index_table = bgmaker.make_total_index_table(
        data_store=data_store,
        modeltype='2D',
        out_dir_background_model=outdir2,
        filename_obs_group_table=fn
    )

    fn = outdir + '/hdu-index.fits.gz'
    hdu_index_table.write(fn, overwrite=True)

    offset_band = Angle([0, 2.49], 'deg')

    ref_cube_images = make_empty_cube(emin=0.5, emax=100, enumbins=5)
    ref_cube_exposure = make_empty_cube(emin=0.1, emax=120, enumbins=80, data_unit="m2 s")
    ref_cube_skymask = make_empty_cube(emin=0.5, emax=100, enumbins=5)

    data_store = DataStore.from_dir(tmpdir)

    refheader = ref_cube_images.sky_image_ref.to_image_hdu().header
    exclusion_mask = SkyImage.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    ref_cube_skymask.data = np.tile(exclusion_mask.data, (5, 1, 1))

    # TODO: Problem with the load psftable for one of the run that is not implemented yet...
    data_store.hdu_table.remove_row(14)

    # Cube Analysis
    cube_maker = StackedObsCubeMaker(
        empty_cube_images=ref_cube_images, empty_exposure_cube=ref_cube_exposure,
        offset_band=offset_band, data_store=data_store, obs_table=data_store.obs_table,
        exclusion_mask=ref_cube_skymask, save_bkg_scale=True,
    )
    cube_maker.make_cubes(make_background_image=True, radius=10.)

    assert_allclose(cube_maker.counts_cube.data.sum(), 4898.0, atol=3)
    assert_allclose(cube_maker.bkg_cube.data.sum(), 4260.120595293951, atol=3)

    # Note: the tolerance in the following assert is low to pass here:
    # https://travis-ci.org/gammapy/gammapy/jobs/234062946#L2112

    cube_maker.significance_cube.data[np.where(np.isinf(cube_maker.significance_cube.data))] = 0
    actual = np.nansum(cube_maker.significance_cube.data)
    assert_allclose(actual, 65777.69960178432, rtol=0.1)

    actual = cube_maker.excess_cube.data.sum()
    assert_allclose(actual, 637.8794047060486, rtol=1e-2)

    actual = np.nansum(cube_maker.exposure_cube.data.to('m2 s').value)
    assert_allclose(actual, 5399539029926424.0, rtol=1e-2)

    assert_allclose(cube_maker.table_bkg_scale[0]["bkg_scale"], 0.8996676356375191, rtol=0.03)

    assert len(cube_maker.counts_cube.energies()) == 5
    assert len(cube_maker.bkg_cube.energies()) == 5
    assert len(cube_maker.significance_cube.energies()) == 5
    assert len(cube_maker.excess_cube.energies()) == 5
    assert len(cube_maker.exposure_cube.energies()) == 80
