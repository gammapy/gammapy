"""Example how to make a Cube analysis from a 2D background model.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Quantity
from ...extern.pathlib import Path
from ...utils.testing import requires_dependency, requires_data, pytest
from ...utils.energy import Energy
from ...data import DataStore
from ...image import SkyImage
from ...background import OffDataBackgroundMaker
from .. import StackedObsCubeMaker
from .. import SkyCube


def make_empty_cube(image_size, energy, center, data_unit=None):
    """
    Make an empty `SkyCube` from a given `SkyImage` and an energy binning.

    Parameters
    ----------
    image_size:int, Total number of pixel of the 2D map
    energy: Tuple for the energy axis: (Emin,Emax,nbins)
    center: SkyCoord of the source
    data_unit : str, Data unit.
    """
    def_image = dict()
    def_image["nxpix"] = image_size
    def_image["nypix"] = image_size
    def_image["binsz"] = 0.02
    def_image["xref"] = center.galactic.l.deg
    def_image["yref"] = center.galactic.b.deg
    def_image["proj"] = 'TAN'
    def_image["coordsys"] = 'GAL'
    def_image["unit"] = data_unit
    e_min, e_max, nbins = energy
    return SkyCube.empty(
        emin=e_min.value, emax=e_max.value, enumbins=nbins, eunit=e_min.unit,
        mode='edges', **def_image
    )


# Temp xfail for this: https://github.com/gammapy/gammapy/pull/899#issuecomment-281001655
@pytest.mark.xfail
@requires_dependency('reproject')
@requires_data('gammapy-extra')
def test_cube_pipe(tmpdir):
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

    center = SkyCoord(83.63, 22.01, unit='deg').galactic
    offset_band = Angle([0, 2.49], 'deg')

    ref_cube_images = make_empty_cube(image_size=250, energy=[Energy(0.5, "TeV"), Energy(100, "TeV"), 5], center=center)
    ref_cube_exposure = make_empty_cube(image_size=250, energy=[Energy(0.1, "TeV"), Energy(120, "TeV"), 80],
                                        center=center, data_unit="m2 s")
    ref_cube_skymask = make_empty_cube(image_size=250, energy=[Energy(0.5, "TeV"), Energy(100, "TeV"), 5],
                                       center=center)

    data_store = DataStore.from_dir(tmpdir)

    refheader = ref_cube_images.sky_image_ref.to_image_hdu().header
    exclusion_mask = SkyImage.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    exclusion_mask = exclusion_mask.reproject(reference=refheader)
    ref_cube_skymask.data = np.tile(exclusion_mask.data, (5, 1, 1))
    # Pb with the load psftable for one of the run that is not implemented yet...
    data_store.hdu_table.remove_row(14)

    # Cube Analysis
    cube_maker = StackedObsCubeMaker(empty_cube_images=ref_cube_images, empty_exposure_cube=ref_cube_exposure,
                                     offset_band=offset_band, data_store=data_store, obs_table=data_store.obs_table,
                                     exclusion_mask=ref_cube_skymask, save_bkg_scale=True)
    cube_maker.make_cubes(make_background_image=True, radius=10.)

    assert_allclose(cube_maker.counts_cube.data.sum(), 4898.0, atol=3)
    assert_allclose(cube_maker.bkg_cube.data.sum(), 4260.120595293951, atol=3)
    cube_maker.significance_cube.data[np.where(np.isinf(cube_maker.significance_cube.data))] = 0
    assert_allclose(np.nansum(cube_maker.significance_cube.data), 67613.24519908393, atol=3)
    assert_allclose(cube_maker.excess_cube.data.sum(), 637.8794047060486, atol=3)
    assert_quantity_allclose(np.nansum(cube_maker.exposure_cube.data), Quantity(4891844242766714.0, "m2 s"),
                             atol=Quantity(3, "m2 s"))
    assert_allclose(cube_maker.table_bkg_scale[0]["bkg_scale"], 0.8956177614218819)

    assert len(cube_maker.counts_cube.energies()) == 5
    assert len(cube_maker.bkg_cube.energies()) == 5
    assert len(cube_maker.significance_cube.energies()) == 5
    assert len(cube_maker.excess_cube.energies()) == 5
    assert len(cube_maker.exposure_cube.energies()) == 80
