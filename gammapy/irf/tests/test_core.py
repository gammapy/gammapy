# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u
from gammapy.irf.core import IRF, FoVAlignment
from gammapy.maps import MapAxis


class TestIRF(IRF):
    tag = "myirf"
    required_axes = ["energy", "offset"]
    default_unit = u.deg


def test_irf_init_quantity():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1)

    irf = TestIRF(axes=[energy_axis, offset_axis], data=data, unit=u.deg)
    irf2 = TestIRF(axes=[energy_axis, offset_axis], data=data * u.deg)

    assert np.all(irf.quantity == irf2.quantity)


def test_immutable():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1) * u.deg
    test_irf = TestIRF(
        axes=[energy_axis, offset_axis],
        data=data,
        is_pointlike=False,
        fov_alignment=FoVAlignment.RADEC,
    )

    with pytest.raises(AttributeError):
        test_irf.is_pointlike = True

    with pytest.raises(AttributeError):
        test_irf.fov_alignment = FoVAlignment.ALTAZ
