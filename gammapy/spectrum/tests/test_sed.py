# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...datasets import FermiGalacticCenter
from ...image import lon_lat_rectangle_mask
from ...spectrum import SED, add_spec, cube_sed
from ...spectrum.models import PowerLaw, PLExpCutoff


@pytest.mark.xfail
def test_add_spec():
    # Create empty frame
    sed = SED()
    # Add one spectrum
    model = [PowerLaw, [1e-10, 2.2], [1e3]]
    xlim = [1e2, 1e4]
    add_spec(sed.ax, model, xlim)
    # Add another spectrum
    model = [PLExpCutoff, [1e-10, 2.0, 1e3], [1e3]]
    xlim = [1e1, 1e4]
    add_spec(sed.ax, model, xlim, color='r')
    # Show it
    sed.fig.show()


@pytest.mark.xfail
def test_1():
    """ Plot a test butterfly with hardcoded parameters """

    # Create empty plot frame
    sed = SED()

    # Plot a butterfly in the Fermi SED region
    sed.ecpl_params['e_pivot'] = 1e3
    sed.ecpl_params['e_min'] = 1e2
    sed.ecpl_params['e_max'] = 1e5
    sed.ecpl_params['e_cut'] = 1e4
    sed.ecpl_params['e_cut_err'] = 0
    sed.ecpl_params['e_scale'] = 1e6
    sed.ecpl_params['norm'] = 1e-9
    sed.ecpl_params['norm_err'] = 1e-10
    sed.ecpl_params['norm_scale'] = 1e-6
    sed.ecpl_params['index'] = 2
    sed.ecpl_params['index_err'] = 0.2
    sed.ecpl_params['color'] = 'g'
    sed.ecpl_params['butterfly'] = True
    sed.plot_ecpl(**sed.ecpl_params)

    # Plot a second butterfly in the HESS SED region
    sed.ecpl_params['e_scale'] = 1e6
    sed.ecpl_params['norm_scale'] = 1e-12
    sed.ecpl_params['color'] = 'blue'
    sed.ecpl_params['e_cut'] = 1e100
    sed.plot_ecpl(**sed.ecpl_params)

    # Plot a few flux points and upper limits
    sed.plot_point(1e9, 1e-10)
    sed.plot_point(1e10, 1e-10, ul=True)
    sed.plot_point(1e9, 1e-11, e_err=3e8, f_err=3e-11)
    sed.plot_point(1e12, 1e-10, e_err=[[1e11], [1e13]],
                   ul=True, e_err_abs=True)

    # Save plot in file
    sed.save('../_temp/test_1.pdf')


@pytest.mark.xfail
def test_2():
    """ Plot the Crab nebula as given in the HESS and Fermi catalogs """

    # Catalog input files and plot output file
    cat_dir = "/Users/deil/work/gev_tev_connection/christoph/_catalogs/"
    hess_cat_name = cat_dir + "hess_cat_full.fits"
    fermi_cat_name = cat_dir + "fermi_cat_full.fit"
#    hess_object_name = 'HESS J1713-381'
#    fermi_object_name = '1FGL J0534.5+2200'
    hess_object_name = 'HESS J0534+220'
    fermi_object_name = '1FGL J0534.5+2200'

    # Create empty plot frame
    sed = SED()

    # Plot the Crab SED by looking up parameters from catalogs
    sed.add_component('hess', hess_cat_name, hess_object_name,
                      color='b')
#    sed.add_component('fermi',fermi_cat_name, fermi_object_name,
#                      color = 'g')

    # Save plot in file
    sed.save('../_temp/test_2.pdf')


@pytest.mark.xfail
def test_42():
    """Examples plotted in the 2FGL paper (index starting at 0):
    Figure Index Name                ASSOC1               ASSOC2
    6  -> source name not given in draft
    7      605   2FGL J0007.0+7303   LAT PSR J0007+7303   (Pulsar in CTA1)
    8      586   2FGL J1224.9+2122   4C+21.35
    -      646   2FGL J0835.3-4510   PSR J0835-4510       Vela
    -      637   2FGL J0534.5+2201   PSR J0534+2200       Crab
    """
    sed = SED()
    sed.add(['2FGL J0007.0+7303'])  # CTA1
    # sed.add(['2FGL J1224.9+2122']) # 4C+21.35
    # sed.add(['2FGL J0534.5+2201']) # Crab
    sed.plot('sed.png')

def test_cube_sed1():
    """Tests against known results with differential cube of 1s.
    """
    spec_cube = FermiGalacticCenter.diffuse_model()
    spec_cube.data = 10 * np.ones_like(spec_cube.data)

    counts = FermiGalacticCenter.diffuse_model()
    counts.data = np.ones_like(counts.data)

    lons, lats = spec_cube.spatial_coordinate_images
    
    mask = lon_lat_rectangle_mask(lons.value, lats.value, -8, 8, -4, 4)

    sed_table1 = cube_sed(spec_cube, mask, flux_type='differential')
    assert_allclose(sed_table1['DIFF_FLUX'].data, 2560)# * np.ones(30))
    assert_allclose(sed_table1['DIFF_FLUX_ERR'].data, 0)

    sed_table2 = cube_sed(spec_cube, mask, flux_type='differential',
                          errors=True, standard_error = 0.1)
    assert_allclose(sed_table2['DIFF_FLUX_ERR'].data, 256)

    sed_table3 = cube_sed(spec_cube, mask, flux_type='differential',
                          errors=True, counts = counts)
    assert_allclose(sed_table3['DIFF_FLUX_ERR'].data, 2560 * np.sqrt(1./256))

def test_cube_sed2():
    """Tests against known results with integral cube of 1s.
    """
    spec_cube = FermiGalacticCenter.diffuse_model()
    spec_cube.data = 10 * np.ones_like(spec_cube.data[:-1])

    counts = FermiGalacticCenter.diffuse_model()
    counts.data = np.ones_like(counts.data[:-1])

    lons, lats = spec_cube.spatial_coordinate_images

    mask = lon_lat_rectangle_mask(lons.value, lats.value, -8, 8, -4, 4)

    sed_table1 = cube_sed(spec_cube, mask, flux_type='integral')

    assert_allclose(sed_table1['ENERGY'][0], 56.95239033587774)
    assert_allclose(sed_table1['DIFF_FLUX'][0], 170.86224025271986)
    assert_allclose(sed_table1['DIFF_FLUX_ERR'], 0)

    sed_table2 = cube_sed(spec_cube, mask, flux_type='integral',
                          errors=True, standard_error = 0.1)

    assert_allclose(sed_table2['DIFF_FLUX_ERR'][0],
                    0.1 * sed_table2['DIFF_FLUX'][0])

    sed_table3 = cube_sed(spec_cube, mask, flux_type='integral',
                          errors=True, counts = counts)

    assert_allclose(sed_table3['DIFF_FLUX_ERR'][0],
                    np.sqrt(1./256) * sed_table3['DIFF_FLUX'][0])
