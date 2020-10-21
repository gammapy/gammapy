# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.coordinates import SkyCoord
from gammapy.maps import MapAxis, WcsNDMap
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel, PointSpatialModel
from gammapy.estimators import FluxMap


@pytest.fixture(scope="session")
def reference_model():
    return SkyModel(spatial_model=PointSpatialModel(), spectral_model=PowerLawSpectralModel(index=2))

@pytest.fixture(scope="session")
def wcs_flux_map(reference_model):
    energy_axis = MapAxis.from_energy_bounds(0.1,10, 2, unit='TeV')

    map_dict = {}
    map_dict["ref_model"] = reference_model

    map_dict["dnde_ref"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='cm-2 s-1 TeV-1')
    map_dict["dnde_ref"].quantity += reference_model.spectral_model(energy_axis.center)[:,np.newaxis, np.newaxis]

    map_dict["norm"]= WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm"].data += 1.0

    map_dict["norm_err"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_err"].data += 0.1

    map_dict["norm_errp"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_errp"].data += 0.2

    map_dict["norm_errn"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_errn"].data += 0.2

    map_dict["norm_ul"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_ul"].data += 2.0

    return map_dict


def test_flux_map_properties(wcs_flux_map):
    fluxmap = FluxMap(**wcs_flux_map)

    assert_allclose(fluxmap.dnde.data[:,0,0],[1e-11, 1e-13])
    assert_allclose(fluxmap.dnde_err.data[:,0,0],[1e-12, 1e-14])
    assert_allclose(fluxmap.dnde_err.data[:,0,0],[1e-12, 1e-14])
    assert_allclose(fluxmap.dnde_errn.data[:,0,0],[2e-12, 2e-14])
    assert_allclose(fluxmap.dnde_errp.data[:,0,0],[2e-12, 2e-14])
    assert_allclose(fluxmap.dnde_ul.data[:,0,0],[2e-11, 2e-13])

    assert_allclose(fluxmap.flux.data[:,0,0],[9e-12, 9e-13])
    assert_allclose(fluxmap.flux_err.data[:,0,0],[9e-13, 9e-14])
    assert_allclose(fluxmap.flux_errn.data[:,0,0],[18e-13, 18e-14])
    assert_allclose(fluxmap.flux_errp.data[:,0,0],[18e-13, 18e-14])
    assert_allclose(fluxmap.flux_ul.data[:,0,0],[18e-12, 18e-13])

    assert_allclose(fluxmap.eflux.data[:,0,0],[2.302585e-12, 2.302585e-12])
    assert_allclose(fluxmap.eflux_err.data[:,0,0],[2.302585e-13, 2.302585e-13])
    assert_allclose(fluxmap.eflux_errp.data[:,0,0],[4.60517e-13, 4.60517e-13])
    assert_allclose(fluxmap.eflux_errn.data[:,0,0],[4.60517e-13, 4.60517e-13])
    assert_allclose(fluxmap.eflux_ul.data[:,0,0],[4.60517e-12, 4.60517e-12])

    assert_allclose(fluxmap.e2dnde.data[:, 0, 0], [1e-12, 1e-12])
    assert_allclose(fluxmap.e2dnde_err.data[:, 0, 0], [1e-13, 1e-13])
    assert_allclose(fluxmap.e2dnde_errn.data[:, 0, 0], [2e-13, 2e-13])
    assert_allclose(fluxmap.e2dnde_errp.data[:, 0, 0], [2e-13, 2e-13])
    assert_allclose(fluxmap.e2dnde_ul.data[:, 0, 0], [2e-12, 2e-12])

@pytest.mark.parametrize("sed_type", ["likelihood", "dnde", "flux", "eflux", "e2dnde"])
def test_flux_map_read_write(tmp_path, wcs_flux_map, sed_type):
    fluxmap = FluxMap(**wcs_flux_map)

    fluxmap.write(tmp_path / "tmp.fits", sed_type=sed_type)
    new_fluxmap = FluxMap.read(tmp_path / "tmp.fits")

    assert_allclose(new_fluxmap.norm.data[:,0,0], [1, 1])
    assert_allclose(new_fluxmap.norm_err.data[:,0,0], [0.1, 0.1])
    assert_allclose(new_fluxmap.norm_errn.data[:,0,0], [0.2, 0.2])
    assert_allclose(new_fluxmap.norm_ul.data[:,0,0], [2, 2])

def test_get_flux_point(wcs_flux_map):
    fluxmap = FluxMap(**wcs_flux_map)

    coord = SkyCoord(0., 0., unit="deg", frame="galactic")
    fp = fluxmap.get_flux_points(coord)

    assert_allclose(fp.table["e_min"], [0.1, 1.0])
    assert_allclose(fp.table["norm"], [1, 1] )
    assert_allclose(fp.table["norm_err"], [0.1, 0.1] )
    assert_allclose(fp.table["norm_errn"], [0.2, 0.2] )
    assert_allclose(fp.table["norm_errp"], [0.2, 0.2])
    assert_allclose(fp.table["norm_ul"], [2, 2])
