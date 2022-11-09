# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
import astropy.units as u
from gammapy.irf.core import IRF, FoVAlignment
from gammapy.maps import MapAxis


class MyCustomIRF(IRF):
    tag = "myirf"
    required_axes = ["energy", "offset"]
    default_unit = u.deg


def test_irf_init_quantity():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1)

    irf = MyCustomIRF(axes=[energy_axis, offset_axis], data=data, unit=u.deg)
    irf2 = MyCustomIRF(axes=[energy_axis, offset_axis], data=data * u.deg)

    assert np.all(irf.quantity == irf2.quantity)


def test_immutable():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1) * u.deg
    test_irf = MyCustomIRF(
        axes=[energy_axis, offset_axis],
        data=data,
        is_pointlike=False,
        fov_alignment=FoVAlignment.RADEC,
    )

    with pytest.raises(AttributeError):
        test_irf.is_pointlike = True

    with pytest.raises(AttributeError):
        test_irf.fov_alignment = FoVAlignment.ALTAZ


def test_slice_by_idx():
    energy_axis = MapAxis.from_energy_bounds(10, 100, 10, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 2.5, 5, unit="deg", name="offset")
    data = np.full((10, 5), 1)

    irf = MyCustomIRF(axes=[energy_axis, offset_axis], data=data, unit=u.deg)

    irf_sliced = irf.slice_by_idx({"energy": slice(3, 7)})
    assert irf_sliced.data.shape == (4, 5)
    assert irf_sliced.axes["energy"].nbin == 4

    irf_sliced = irf.slice_by_idx({"offset": slice(3, 5)})
    assert irf_sliced.data.shape == (10, 2)
    assert irf_sliced.axes["offset"].nbin == 2

    irf_sliced = irf.slice_by_idx({"energy": slice(3, 7), "offset": slice(3, 5)})
    assert irf_sliced.data.shape == (4, 2)
    assert irf_sliced.axes["offset"].nbin == 2
    assert irf_sliced.axes["energy"].nbin == 4

    with pytest.raises(ValueError) as exc_info:
        _ = irf.slice_by_idx({"energy": 3, "offset": 7})

    assert (
        str(exc_info.value)
        == "Integer indexing not supported, got {'energy': 3, 'offset': 7}"
    )
