# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose
from gammapy.maps import MapAxis, WcsGeom

asdf = pytest.importorskip("asdf")
pytest.importorskip("asdf.testing")
from asdf.testing.helpers import yaml_to_asdf  # noqa: E402

axes1 = [MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy")]
axes2 = [
    MapAxis(np.logspace(0.0, 3.0, 3), interp="log", name="energy"),
    MapAxis(np.logspace(1.0, 3.0, 4), interp="lin"),
]
skydir = SkyCoord(110.0, 75.0, unit="deg", frame="icrs")

tested_wcs_geom = [
    (None, 10.0, "galactic", "AIT", skydir, None),
    (None, 10.0, "galactic", "AIT", skydir, axes1),
    (None, [10.0, 20.0], "galactic", "AIT", skydir, axes1),
    (None, 10.0, "galactic", "AIT", skydir, axes2),
    (None, [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], "galactic", "AIT", skydir, axes2),
    (10, 0.1, "galactic", "AIT", skydir, None),
    (10, 0.1, "galactic", "AIT", skydir, axes1),
    (10, [0.1, 0.2], "galactic", "AIT", skydir, axes1),
]


@pytest.mark.parametrize(
    ("npix", "binsz", "frame", "proj", "skydir", "axes"), tested_wcs_geom
)
def test_wcsgeom_roundtrip(npix, binsz, frame, proj, skydir, axes, tmp_path):
    file_path = tmp_path / "test.asdf"
    geom = WcsGeom.create(
        npix=npix,
        binsz=binsz,
        frame=frame,
        proj=proj,
        skydir=skydir,
        axes=axes,
    )
    with asdf.AsdfFile() as af:
        af["geom"] = geom
        af.write_to(file_path)
    with asdf.open(file_path) as af:
        result = af["geom"]
        assert result.is_allclose(geom)
        if not geom.is_regular:
            assert_allclose(result._cdelt[0], geom._cdelt[0])
            assert_allclose(result._cdelt[1], geom._cdelt[1])
            assert_allclose(result._crpix[0], geom._crpix[0])
            assert_allclose(result._crpix[1], geom._crpix[1])


def test_wcs_geom_invalid():
    example = """!<asdf://gammapy.org/gammapy/tags/maps/wcsgeom-1.0.0>
            npix:
              - !core/ndarray-1.1.0
                data: [50]
              - !core/ndarray-1.1.0
                data: [100]
             """

    buff = yaml_to_asdf(f"example: {example.strip()}")
    with pytest.raises(asdf.exceptions.ValidationError):
        asdf.open(buff)
