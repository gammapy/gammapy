# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from gammapy.irf.core import IRF
from gammapy.maps import MapAxis


def test_irf_init_quantity():
    class TestIRF(IRF):
        tag = "myirf"
        required_axes = ["energy", "offset"]

    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1)

    irf = TestIRF(axes=[energy_axis, offset_axis], data=data, unit=u.deg)
    irf2 = TestIRF(axes=[energy_axis, offset_axis], data=data * u.deg)

    assert np.all(irf.quantity == irf2.quantity)
