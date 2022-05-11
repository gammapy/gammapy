import numpy as np
import astropy.units as u


def test_effective_area_2d_to_gadf():
    from gammapy.irf import EffectiveAreaTable2D
    from gammapy.maps import MapAxis

    energy_axis = MapAxis.from_energy_bounds(
        1 * u.TeV, 10 * u.TeV, nbin=3, name="energy_true"
    )
    offset_axis = MapAxis.from_bounds(0 * u.deg, 2 * u.deg, nbin=2, name="offset")
    data = np.ones((energy_axis.nbin, offset_axis.nbin)) * u.m**2

    aeff = EffectiveAreaTable2D(data=data, axes=[energy_axis, offset_axis])
    hdu = aeff.to_table_hdu(format="gadf-dl3")

    columns = {column.name for column in hdu.columns}
    mandatory_columns = {
        "ENERG_LO",
        "ENERG_HI",
        "THETA_LO",
        "THETA_HI",
        "EFFAREA",
    }

    missing = mandatory_columns.difference(columns)
    assert len(missing) == 0, f"GADF HDU missing required column(s) {missing}"

    header = hdu.header
    assert header["HDUCLASS"] == "GADF"
    assert (
        header["HDUDOC"]
        == "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    )
    assert header["HDUVERS"] == "0.2"
    assert header["HDUCLAS1"] == "RESPONSE"
    assert header["HDUCLAS2"] == "EFF_AREA"
    assert header["HDUCLAS3"] == "FULL-ENCLOSURE"
    assert header["HDUCLAS4"] == "AEFF_2D"
    assert header["EXTNAME"] == "EFFECTIVE AREA"


def test_energy_dispersion_2d_to_gadf():
    from gammapy.irf import EnergyDispersion2D
    from gammapy.maps import MapAxis

    energy_axis = MapAxis.from_energy_bounds(
        1 * u.TeV, 10 * u.TeV, nbin=3, name="energy_true"
    )
    offset_axis = MapAxis.from_bounds(0 * u.deg, 2 * u.deg, nbin=2, name="offset")
    migra_axis = MapAxis.from_bounds(0.2, 5, nbin=5, interp="log", name="migra")
    data = np.zeros((energy_axis.nbin, migra_axis.nbin, offset_axis.nbin))

    edisp = EnergyDispersion2D(data=data, axes=[energy_axis, migra_axis, offset_axis])
    hdu = edisp.to_table_hdu(format="gadf-dl3")

    mandatory_columns = {
        "ENERG_LO",
        "ENERG_HI",
        "MIGRA_LO",
        "MIGRA_HI",
        "THETA_LO",
        "THETA_HI",
        "MATRIX",
    }
    columns = {column.name for column in hdu.columns}
    missing = mandatory_columns.difference(columns)
    assert len(missing) == 0, f"GADF HDU missing required column(s) {missing}"

    header = hdu.header
    assert header["HDUCLASS"] == "GADF"
    assert (
        header["HDUDOC"]
        == "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    )
    assert header["HDUVERS"] == "0.2"
    assert header["HDUCLAS1"] == "RESPONSE"
    assert header["HDUCLAS2"] == "EDISP"
    assert header["HDUCLAS3"] == "FULL-ENCLOSURE"
    assert header["HDUCLAS4"] == "EDISP_2D"


def test_psf_3d_to_gadf():
    from gammapy.irf import PSF3D
    from gammapy.maps import MapAxis

    energy_axis = MapAxis.from_energy_bounds(
        1 * u.TeV, 10 * u.TeV, nbin=3, name="energy_true"
    )
    offset_axis = MapAxis.from_bounds(0 * u.deg, 2 * u.deg, nbin=2, name="offset")
    rad_axis = MapAxis.from_bounds(0.0 * u.deg, 1 * u.deg, nbin=10, name="rad")
    data = np.zeros((energy_axis.nbin, offset_axis.nbin, rad_axis.nbin)) / u.sr

    psf = PSF3D(data=data, axes=[energy_axis, offset_axis, rad_axis])
    hdu = psf.to_table_hdu(format="gadf-dl3")

    mandatory_columns = {
        "ENERG_LO",
        "ENERG_HI",
        "THETA_LO",
        "THETA_HI",
        "RAD_LO",
        "RAD_HI",
        "RPSF",
    }
    columns = {column.name for column in hdu.columns}
    missing = mandatory_columns.difference(columns)
    assert len(missing) == 0, f"GADF HDU missing required column(s) {missing}"

    header = hdu.header
    assert header["HDUCLASS"] == "GADF"
    assert (
        header["HDUDOC"]
        == "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats"
    )
    assert header["HDUVERS"] == "0.2"
    assert header["HDUCLAS1"] == "RESPONSE"
    assert header["HDUCLAS2"] == "RPSF"
    assert header["HDUCLAS3"] == "FULL-ENCLOSURE"
    assert header["HDUCLAS4"] == "PSF_TABLE"
