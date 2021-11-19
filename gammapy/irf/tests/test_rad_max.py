import numpy as np
import astropy.units as u
from gammapy.maps import MapAxis


def test_rad_max_roundtrip(tmp_path):
    from gammapy.irf import RadMax2D

    n_energy = 10
    energy_axis = MapAxis.from_energy_bounds(
        50 * u.GeV, 100 * u.TeV, n_energy, name="energy"
    )

    n_offset = 5
    offset_axis = MapAxis.from_bounds(0, 2, n_offset, unit=u.deg, name="offset")

    shape = (n_energy, n_offset)
    rad_max = np.linspace(0.1, 0.5, n_energy * n_offset).reshape(shape)

    rad_max_2d = RadMax2D(
        axes=[
            energy_axis,
            offset_axis,
        ],
        data=rad_max,
        unit=u.deg,
    )

    rad_max_2d.write(tmp_path / "rad_max.fits")
    rad_max_read = RadMax2D.read(tmp_path / "rad_max.fits")

    assert np.all(rad_max_read.data.data == rad_max)
    assert np.all(rad_max_read.data.data == rad_max_read.data.data)
