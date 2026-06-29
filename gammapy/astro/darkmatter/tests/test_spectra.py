# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import numpy as np
import astropy.units as u
from astropy.table import Table

from gammapy.astro.darkmatter import (
    BoxPrimaryFlux,
    PrimaryFlux,
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    MonochromaticPrimaryFlux,
    VIBPrimaryFlux,
)
from gammapy.modeling.models import Models, SkyModel
from gammapy.utils.testing import assert_quantity_allclose, requires_data


@requires_data()
def test_primary_flux():
    with pytest.raises(ValueError):
        PrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = PrimaryFlux(channel="W", mDM=1 * u.TeV)
    actual = primflux(500 * u.GeV)
    desired = 9.3319318e-05 / u.GeV
    assert_quantity_allclose(actual, desired)


@pytest.mark.parametrize(
    "mass, expected_flux, source, expected_exception",
    [
        (1.6, 0.00025037, "pppc4", None),
        (11, 0.00549445, "cosmixs", None),
        (75, None, "nonexistend", ValueError),
        (75, None, "pppc4", ValueError),
    ],
)
@requires_data()
def test_primary_flux_interpolation(mass, expected_flux, source, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            PrimaryFlux(
                channel="aZ",
                mDM=mass * u.TeV,
                source=source,
                mapping_dict={"mDM": "mDM"},
            )
        return
    primflux = PrimaryFlux(channel="W", mDM=mass * u.TeV, source=source)
    actual = primflux(500 * u.GeV)
    assert_quantity_allclose(actual, expected_flux / u.GeV, rtol=1e-5)


@pytest.mark.parametrize(
    "model_class, jfactor_unit, expected_flux, expected_dnde, source",
    [
        (
            DarkMatterAnnihilationSpectralModel,
            "GeV2 cm-5",
            6.19575457e-14,
            2.97831615e-16,
            None,
        ),
        (
            DarkMatterDecaySpectralModel,
            "GeV cm-2",
            3.209234e-2,
            2.33485775e-05,
            "pppc4",
        ),
        (
            DarkMatterAnnihilationSpectralModel,
            "GeV2 cm-5",
            6.03197683e-14,
            3.52065879e-16,
            "cosmixs",
        ),
        (DarkMatterDecaySpectralModel, "GeV cm-2", 0.031677, 2.771877e-05, "cosmixs"),
    ],
)
@requires_data()
def test_dm_spectral_model(
    tmpdir, jfactor_unit, model_class, expected_flux, expected_dnde, source
):
    channel = "b"
    mass = 5 * u.TeV
    jfactor = 3.41e19 * u.Unit(jfactor_unit)
    energy_min = 0.01 * u.TeV
    energy_max = 10 * u.TeV

    model = model_class(mass=mass, channel=channel, jfactor=jfactor, source=source)
    flux = model.integral(energy_min=energy_min, energy_max=energy_max).to("cm-2 s-1")
    dnde = model.evaluate(energy=1 * u.TeV, scale=1).to("cm-2 s-1 TeV-1")

    sky_model = SkyModel(
        spectral_model=model,
        name="skymodel",
    )
    models = Models([sky_model])
    filename = tmpdir / "model.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)

    assert_quantity_allclose(flux.value, expected_flux, rtol=1e-3)
    assert_quantity_allclose(dnde.value, expected_dnde, rtol=1e-3)

    assert new_models[0].spectral_model.channel == model.channel
    assert new_models[0].spectral_model.z == model.z
    assert_allclose(new_models[0].spectral_model.jfactor.value, model.jfactor.value)
    assert new_models[0].spectral_model.mass.value == 5
    assert new_models[0].spectral_model.mass.unit == u.TeV


@requires_data()
def test_primary_flux_cosmixs():
    with pytest.raises(ValueError):
        PrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = PrimaryFlux(
        channel="b", mDM=1 * u.TeV, source="cosmixs", mapping_dict={"mDM": "mDM"}
    )
    actual = primflux(500 * u.GeV)
    desired = 1.842842e-05 / u.GeV
    assert_quantity_allclose(actual, desired, rtol=1e-4)

    with pytest.raises(ValueError):
        PrimaryFlux(channel="q", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="V->e", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="V->mu", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="V->tau", mDM=1 * u.TeV, source="cosmixs")

    with pytest.raises(ValueError):
        PrimaryFlux(channel="d", mDM=1 * u.TeV, source="pppc4")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="u", mDM=1 * u.TeV, source="pppc4")
    with pytest.raises(ValueError):
        PrimaryFlux(channel="s", mDM=1 * u.TeV, source="pppc4")


@requires_data()
def test_primary_flux_cosmixs_channel_q_error():
    """Covers the branch raising ValueError for channel='q' with source='cosmixs'."""
    with pytest.raises(ValueError, match="The channel q is not available in cosmixs"):
        PrimaryFlux(channel="q", mDM=1 * u.TeV, source="cosmixs")


def test_custom_source_file_empty(tmp_path):
    """Test that an empty custom source file raises the correct error."""
    empty_file = tmp_path / "empty_spectra.dat"
    empty_file.touch()

    with pytest.raises(KeyError, match="Source file is empty."):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", source=str(empty_file)
        )


def test_custom_source_file_unrecognized_format(tmp_path):
    """A file with content that Table.read cannot parse should raise a
    clear error, regardless of its extension (extensions are no longer
    whitelisted; Table.read decides what is readable)."""
    bad_file = tmp_path / "unreadable_spectra.dl2"
    bad_file.write_text("this is not a valid table format §§§ %%% ???")

    with pytest.raises(Exception):
        DarkMatterAnnihilationSpectralModel(
            mass=5 * u.TeV, channel="b", source=str(bad_file)
        )


def test_custom_source_fits_extension_supported(tmp_path):
    """A .fits file should be readable as a custom source (previously
    excluded by the removed extension whitelist)."""
    fits_file = tmp_path / "custom_spectra.fits"
    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0, -3.0, -2.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16],
        }
    )
    t.write(fits_file, format="fits")

    model = DarkMatterAnnihilationSpectralModel(
        mass=500 * u.GeV, channel="b", source=str(fits_file)
    )
    assert model.channel == "b"


def test_dm_source_as_table():
    """Test that a source can be passed directly as an astropy.table.Table."""
    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0, -3.0, -2.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16],
        }
    )

    model = DarkMatterAnnihilationSpectralModel(mass=500 * u.GeV, channel="b", source=t)
    assert model.channel == "b"
    assert isinstance(model.source, Table)

    dnde = model.evaluate(energy=200 * u.GeV, scale=1)
    assert dnde.value >= 0
    assert np.isfinite(dnde.value)


def test_dm_source_as_table_with_mapping():
    """Test that a source Table works together with a mapping_dict, exactly
    like a custom file source does."""
    t = Table(
        {
            "mass": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "energy": [-3.0, -2.0, -3.0, -2.0],
            "bbar": [1e-15, 1e-16, 1e-15, 1e-16] / u.GeV,
        }
    )
    mapping = {"energy": "Log[10,x]", "mass": "mDM", "bbar": "b"}

    model = DarkMatterAnnihilationSpectralModel(
        mass=500 * u.GeV, channel="b", source=t, mapping_dict=mapping
    )
    assert model.channel == "b"
    assert model.mapping_dict == mapping


def test_dm_source_as_table_missing_channel():
    """A Table source missing the requested channel column should raise,
    just like a custom file source does."""
    t = Table(
        {
            "mDM": [500.0, 500.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0],
            "b": [1e-15, 1e-16],
        }
    )

    with pytest.raises(ValueError, match="The channel eL is not available"):
        DarkMatterAnnihilationSpectralModel(mass=500 * u.GeV, channel="eL", source=t)


def test_dm_source_invalid_type():
    """source must be None, a string, or a Table -- anything else should
    raise a clear TypeError."""
    with pytest.raises(TypeError, match="source must be"):
        DarkMatterAnnihilationSpectralModel(mass=500 * u.GeV, channel="b", source=12345)

    with pytest.raises(TypeError, match="source must be"):
        DarkMatterAnnihilationSpectralModel(
            mass=500 * u.GeV, channel="b", source=["not", "a", "table"]
        )


def test_dm_spectral_model_custom_io(tmp_path):
    """Test that source file and mapping_dict survive YAML serialization."""
    custom_file = tmp_path / "custom_spectra.ecsv"

    t = Table(
        {
            "mass": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "energy": [100.0, 200.0, 100.0, 200.0] * u.GeV,
            "b": [1e-15, 1e-16, 1e-15, 1e-16] / u.GeV,
        }
    )
    t.write(custom_file, format="ascii.ecsv")

    mapping = {"energy": "Log[10,x]", "mass": "mDM", "b": "b"}

    model = DarkMatterAnnihilationSpectralModel(
        mass=500 * u.GeV,
        channel="b",
        jfactor=3.41e19 * u.Unit("GeV2 cm-5"),
        source=str(custom_file),
        mapping_dict=mapping,
    )

    sky_model = SkyModel(spectral_model=model, name="skymodel_custom")
    models = Models([sky_model])

    filename = tmp_path / "model_custom.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)
    loaded_model = new_models[0].spectral_model

    assert loaded_model.source == str(custom_file)
    assert loaded_model.mapping_dict == mapping


def test_dm_annihilation_custom_errors(tmp_path):
    file_path = tmp_path / "test_dm.ecsv"
    t = Table()
    t["mDM"] = [1000, 5000] * u.GeV
    t["Log[10,x]"] = [-3, -2]
    t["bbar"] = [1e-5, 1e-4]
    t.write(file_path)

    mass = 5 * u.TeV
    with pytest.raises(TypeError, match="mapping_dict must be a dictionary"):
        DarkMatterAnnihilationSpectralModel(
            mass=mass,
            channel="b",
            source=str(file_path),
            mapping_dict=["not", "a", "dict"],
        )

    incomplete_mapping = {"Log[10,x]": "Log[10,x]"}
    with pytest.raises(KeyError, match="Mandatory column"):
        DarkMatterAnnihilationSpectralModel(
            mass=mass,
            channel="b",
            source=str(file_path),
            mapping_dict=incomplete_mapping,
        )

    wrong_mapping = {"mDM": "mDM", "Log[10,x]": "Log[10,x]", "wrong_col": "tau"}
    with pytest.raises(
        ValueError,
        match="The channel b is not available \
                            in the provided mapping dictionary. Please choose another \
                            channel or check the mapping_dict provided.\n",
    ):
        DarkMatterAnnihilationSpectralModel(
            mass=mass, channel="b", source=str(file_path), mapping_dict=wrong_mapping
        )


def test_missing_data_file(monkeypatch):
    """Test that FileNotFoundError is raised if the data path does not exist."""
    monkeypatch.setenv("GAMMAPY_DATA", "/fake/path/to/nowhere")
    with pytest.raises(FileNotFoundError, match="File not found"):
        PrimaryFlux(mDM=1 * u.TeV, channel="b")


@requires_data()
def test_mDM_out_of_bounds():
    """Test that ValueError is raised if the mass is out of the interpolation table bounds."""
    with pytest.raises(ValueError, match="is out of the bounds of the model"):
        PrimaryFlux(mDM=1 * u.eV, channel="b")


def test_custom_source_file_without_mapping_and_missing_channel(tmp_path):
    """Test the use of custom files without mapping_dict and with missing channels."""
    custom_file = tmp_path / "custom_spectra_nomap.ecsv"

    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [100.0, 200.0, 100.0, 200.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16],
        }
    )
    t.write(custom_file, format="ascii.ecsv")

    model = DarkMatterAnnihilationSpectralModel(
        mass=500 * u.GeV, channel="b", source=str(custom_file)
    )
    assert model.channel == "b"

    with pytest.raises(ValueError, match="The channel eL is not available"):
        DarkMatterAnnihilationSpectralModel(
            mass=500 * u.GeV, channel="eL", source=str(custom_file)
        )


def test_dm_annihilation_with_monochromatic_primary_flux():
    """A two-photon line as primary_flux integrates to a sharp spectrum."""
    mDM = 1 * u.TeV
    mono_flux = MonochromaticPrimaryFlux(mDM=mDM, n_gamma_photons=2)

    model = DarkMatterAnnihilationSpectralModel(
        mDM=mDM, channel="b", primary_flux=mono_flux, factor=1 * u.Unit("GeV2 cm-5")
    )

    dnde = model.evaluate(energy=mDM, scale=1)
    assert dnde.value > 0

    data = model.to_dict()
    new_model = DarkMatterAnnihilationSpectralModel.from_dict(data)
    assert isinstance(new_model.primary_flux, MonochromaticPrimaryFlux)
    assert new_model.primary_flux.n_gamma_photons == 2


@requires_data()
def test_dm_annihilation_with_box_primary_flux():
    mDM = 1 * u.TeV
    box_flux = BoxPrimaryFlux(mDM=mDM, mPhi=[100, 50] * u.GeV)

    model = DarkMatterAnnihilationSpectralModel(
        mDM=mDM, channel="b", primary_flux=box_flux, factor=1 * u.Unit("GeV2 cm-5")
    )

    data = model.to_dict()
    new_model = DarkMatterAnnihilationSpectralModel.from_dict(data)
    assert isinstance(new_model.primary_flux, BoxPrimaryFlux)
    assert_quantity_allclose(new_model.primary_flux.mPhi1, box_flux.mPhi1)
    assert_quantity_allclose(new_model.primary_flux.mPhi2, box_flux.mPhi2)


@requires_data()
def test_dm_decay_with_vib_primary_flux():
    """VIB primary flux is valid for mDM/2 expected mass in decay."""
    mDM = 2 * u.TeV
    vib_flux = VIBPrimaryFlux(mDM=mDM / 2)

    model = DarkMatterDecaySpectralModel(
        mDM=mDM, channel="b", primary_flux=vib_flux, factor=1 * u.Unit("GeV cm-2")
    )

    data = model.to_dict()
    new_model = DarkMatterDecaySpectralModel.from_dict(data)
    assert isinstance(new_model.primary_flux, VIBPrimaryFlux)
    assert_quantity_allclose(new_model.primary_flux.mDM, vib_flux.mDM)


# ---------------------------------------------------------------------------
# MonochromaticPrimaryFlux
# ---------------------------------------------------------------------------


def test_monochromatic_two_photon():
    """Two-photon line: integral of the Gaussian should equal n_gamma_photons."""
    mDM = 1 * u.TeV
    flux = MonochromaticPrimaryFlux(mDM=mDM, n_gamma_photons=2)

    assert flux.get_line_energy() == mDM

    energy = np.linspace(0.5, 1.5, 20001) * u.TeV
    dnde = flux.evaluate(energy)
    integral = np.trapezoid(dnde.to_value("TeV-1"), energy.to_value("TeV"))
    assert_allclose(integral, 2.0, rtol=1e-3)


def test_monochromatic_one_photon_z_counterpart():
    """One-photon channel with Z counterpart: line energy below mDM."""
    mDM = 1 * u.TeV
    flux = MonochromaticPrimaryFlux(mDM=mDM, n_gamma_photons=1, counterpart="z")

    e0 = flux.get_line_energy()
    expected = mDM * (1 - (91.2 * u.GeV) ** 2 / (4 * mDM**2))
    assert_quantity_allclose(e0, expected)
    assert e0 < mDM
    assert flux.counterpart_mass == 91.2 * u.GeV


def test_monochromatic_one_photon_h_counterpart_default_mass():
    flux = MonochromaticPrimaryFlux(mDM=1 * u.TeV, n_gamma_photons=1, counterpart="h")
    assert flux.counterpart_mass == 125.1 * u.GeV


def test_monochromatic_known_counterpart_explicit_mass_override():
    flux = MonochromaticPrimaryFlux(
        mDM=1 * u.TeV,
        n_gamma_photons=1,
        counterpart="z",
        counterpart_mass=80,
    )
    assert flux.counterpart_mass == 80 * u.GeV


def test_monochromatic_two_photon_counterpart_mass_none():
    flux = MonochromaticPrimaryFlux(mDM=1 * u.TeV, n_gamma_photons=2)
    assert flux.counterpart is None
    assert flux.counterpart_mass is None
    flux.counterpart_mass = None
    assert flux.counterpart_mass is None


def test_monochromatic_one_photon_custom_counterpart_mass():
    flux = MonochromaticPrimaryFlux(
        mDM=1 * u.TeV, n_gamma_photons=1, counterpart="x", counterpart_mass=50
    )
    assert flux.counterpart_mass == 50 * u.GeV


def test_monochromatic_unknown_counterpart_requires_mass():
    with pytest.raises(ValueError, match="its mass must be provided"):
        MonochromaticPrimaryFlux(mDM=1 * u.TeV, n_gamma_photons=1, counterpart="x")


def test_monochromatic_one_photon_requires_counterpart():
    with pytest.raises(ValueError, match="Counterpart particle must be indicated"):
        MonochromaticPrimaryFlux(mDM=1 * u.TeV, n_gamma_photons=1)


def test_monochromatic_invalid_n_gamma_photons():
    with pytest.raises(ValueError, match="must be 1 or 2"):
        MonochromaticPrimaryFlux(mDM=1 * u.TeV, n_gamma_photons=3)


def test_monochromatic_two_photon_with_counterpart_warns():
    with pytest.warns(UserWarning, match="ignored"):
        flux = MonochromaticPrimaryFlux(
            mDM=1 * u.TeV, n_gamma_photons=2, counterpart="z"
        )
    assert flux.counterpart is None
    assert flux.counterpart_mass is None


def test_monochromatic_kinematically_forbidden():
    with pytest.raises(ValueError, match="Kinematically forbidden"):
        MonochromaticPrimaryFlux(mDM=10 * u.GeV, n_gamma_photons=1, counterpart="z")


def test_monochromatic_energy_out_of_range_warns():
    flux = MonochromaticPrimaryFlux(mDM=1 * u.TeV, n_gamma_photons=2)
    energy = np.linspace(1, 10, 50) * u.GeV
    with pytest.warns(UserWarning, match="outside the"):
        flux.evaluate(energy)


def test_monochromatic_custom_sigma_rel():
    flux = MonochromaticPrimaryFlux(mDM=1 * u.TeV, n_gamma_photons=2, sigma_rel=0.05)
    assert flux.sigma_rel == 0.05


def test_monochromatic_to_from_dict_roundtrip():
    flux = MonochromaticPrimaryFlux(
        mDM=1 * u.TeV, n_gamma_photons=1, counterpart="z", sigma_rel=0.02
    )
    data = flux.to_dict()
    new_flux = MonochromaticPrimaryFlux.from_dict(data)

    assert_quantity_allclose(new_flux.mDM, flux.mDM)
    assert new_flux.n_gamma_photons == flux.n_gamma_photons
    assert new_flux.counterpart == flux.counterpart
    assert_quantity_allclose(new_flux.counterpart_mass, flux.counterpart_mass)
    assert new_flux.sigma_rel == flux.sigma_rel


# ---------------------------------------------------------------------------
# VIBPrimaryFlux
# ---------------------------------------------------------------------------


def test_vib_zero_outside_range():
    """VIB spectrum must vanish for x <= 0 or x >= 1."""
    mDM = 1 * u.TeV
    flux = VIBPrimaryFlux(mDM=mDM)

    energy = u.Quantity([0.0, 1.0, 1.5], "TeV")
    dnde = flux.evaluate(energy)
    assert_allclose(dnde.to_value("TeV-1"), [0.0, 0.0, 0.0])


def test_vib_positive_in_range():
    mDM = 1 * u.TeV
    flux = VIBPrimaryFlux(mDM=mDM)

    energy = np.linspace(0.01, 0.99, 50) * u.TeV
    dnde = flux.evaluate(energy)
    assert np.all(dnde.to_value("TeV-1") >= 0)


def test_vib_invalid_mass():
    with pytest.raises(ValueError, match="must be strictly positive"):
        VIBPrimaryFlux(mDM=-1 * u.TeV)


def test_vib_to_from_dict_roundtrip():
    flux = VIBPrimaryFlux(mDM=2 * u.TeV)
    data = flux.to_dict()
    new_flux = VIBPrimaryFlux.from_dict(data)
    assert_quantity_allclose(new_flux.mDM, flux.mDM)


# ---------------------------------------------------------------------------
# BoxPrimaryFlux
# ---------------------------------------------------------------------------


def test_box_single_mass_normalization():
    """Single mPhi: integral over the box must equal 2 (two photons)."""
    mDM = 1 * u.TeV
    mPhi = [100] * u.GeV
    flux = BoxPrimaryFlux(mDM=mDM, mPhi=mPhi)

    E_phi1, E_phi2 = flux.energies_phi
    assert_quantity_allclose(E_phi1, E_phi2)

    delta_E = flux.delta_E
    e_min = (E_phi1 - delta_E) / 2
    e_max = (E_phi1 + delta_E) / 2

    energy = np.linspace(e_min.to_value("GeV"), e_max.to_value("GeV"), 5000) * u.GeV
    dnde = flux.evaluate(energy)
    integral = np.trapezoid(dnde.to_value("GeV-1"), energy.to_value("GeV"))
    assert_allclose(integral, 2.0, rtol=1e-2)


def test_box_two_distinct_masses():
    """Two distinct masses produce two distinct box centers."""
    mDM = 1 * u.TeV
    mPhi = [100, 50] * u.GeV
    flux = BoxPrimaryFlux(mDM=mDM, mPhi=mPhi)

    E_phi1, E_phi2 = flux.energies_phi
    assert not u.allclose(E_phi1, E_phi2)
    assert flux.mPhi1 == 100 * u.GeV
    assert flux.mPhi2 == 50 * u.GeV


def test_box_two_distinct_masses_overlap_warns():
    """Two close intermediate masses produce overlapping boxes with a warning."""
    mDM = 1 * u.TeV
    mPhi = [100, 99] * u.GeV
    flux = BoxPrimaryFlux(mDM=mDM, mPhi=mPhi)

    energy = np.linspace(400, 500, 100) * u.GeV
    with pytest.warns(UserWarning, match="overlap"):
        flux.evaluate(energy)


def test_box_kinematically_forbidden():
    with pytest.raises(ValueError, match="Kinematically forbidden"):
        BoxPrimaryFlux(mDM=10 * u.GeV, mPhi=[100] * u.GeV)


def test_box_mphi_getter():
    flux = BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=[100, 50] * u.GeV)
    assert flux.mPhi is not None


def test_box_invalid_mphi_length():
    with pytest.raises(ValueError, match="exactly 1 or 2 values"):
        BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=[10, 20, 30] * u.GeV)


def test_box_mphi_setter_none():
    flux = BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=[100] * u.GeV)
    flux.mPhi = None
    assert flux.mPhi is None
    assert flux.mPhi1 is None
    assert flux.mPhi2 is None


def test_box_negative_mphi():
    with pytest.raises(ValueError, match="must be strictly positive"):
        BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=[-10] * u.GeV)


def test_box_to_from_dict_roundtrip():
    flux = BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=[100, 50] * u.GeV)
    data = flux.to_dict()
    new_flux = BoxPrimaryFlux.from_dict(data)

    assert_quantity_allclose(new_flux.mDM, flux.mDM)
    assert_quantity_allclose(new_flux.mPhi1, flux.mPhi1)
    assert_quantity_allclose(new_flux.mPhi2, flux.mPhi2)


def test_box_single_scalar_mphi():
    flux = BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=100 * u.GeV)
    assert flux.mPhi1 == 100 * u.GeV
    assert flux.mPhi2 == 100 * u.GeV


def test_box_evaluate_scalar_energy():
    flux = BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=[100] * u.GeV)
    # Si da error aquí, hay que cambiar len(energy) por np.atleast_1d
    dnde = flux.evaluate(np.array([500]) * u.GeV)
    assert dnde.shape == (1,)


# ---------------------------------------------------------------------------
# _DarkMatterMassValidator / mDM validation
# ---------------------------------------------------------------------------


def test_mDM_wrong_units_raises():
    with pytest.raises(u.UnitConversionError, match="mDM must have energy units"):
        VIBPrimaryFlux(mDM=1 * u.m)


def test_mDM_zero_or_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        VIBPrimaryFlux(mDM=0 * u.GeV)


@requires_data()
def test_dm_annihilation_monochromatic_evaluation_on_dataset():
    """MonochromaticPrimaryFlux set on a MapDataset must produce finite positive flux."""
    from gammapy.datasets import MapDataset
    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.modeling.models import PointSpatialModel

    energy_axis = MapAxis.from_edges(
        [0.01, 0.1, 1.0], unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.1, width=(1, 1), frame="galactic", axes=[energy_axis]
    )
    mDM = 1 * u.TeV
    mono_flux = MonochromaticPrimaryFlux(mDM=mDM, n_gamma_photons=2)
    spectral_model = DarkMatterAnnihilationSpectralModel(
        mDM=mDM,
        channel="b",
        factor=3.41e19 * u.Unit("GeV2 cm-5"),
        primary_flux=mono_flux,
    )
    sky_model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic"),
        name="dm_mono",
    )
    dataset = MapDataset.create(geom, name="test_mono")
    dataset.models = [sky_model]

    assert np.all(np.isfinite(dataset.npred().data))

    # energies around the line energy (= mDM for two-photon)
    flux = spectral_model([0.9, 1.0, 1.1] * u.TeV)
    assert np.all(np.isfinite(flux.value))
    assert np.any(flux.value > 0)


def test_dm_annihilation_vib_evaluation_on_dataset():
    """VIBPrimaryFlux set on a MapDataset must produce finite positive flux."""
    from gammapy.datasets import MapDataset
    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.modeling.models import PointSpatialModel

    energy_axis = MapAxis.from_edges(
        [0.01, 0.1, 1.0], unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.1, width=(1, 1), frame="galactic", axes=[energy_axis]
    )
    mDM = 1 * u.TeV
    vib_flux = VIBPrimaryFlux(mDM=mDM)
    spectral_model = DarkMatterAnnihilationSpectralModel(
        mDM=mDM,
        channel="b",
        factor=3.41e19 * u.Unit("GeV2 cm-5"),
        primary_flux=vib_flux,
    )
    sky_model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic"),
        name="dm_vib",
    )
    dataset = MapDataset.create(geom, name="test_vib")
    dataset.models = [sky_model]

    assert np.all(np.isfinite(dataset.npred().data))

    # VIB is only non-zero for 0 < E < mDM
    flux = spectral_model([0.1, 0.5, 0.9] * u.TeV)
    assert np.all(np.isfinite(flux.value))
    assert np.all(flux.value > 0)


def test_dm_annihilation_box_evaluation_on_dataset():
    """BoxPrimaryFlux set on a MapDataset must produce finite positive flux."""
    from gammapy.datasets import MapDataset
    from gammapy.maps import MapAxis, WcsGeom
    from gammapy.modeling.models import PointSpatialModel

    energy_axis = MapAxis.from_edges(
        [0.01, 0.1, 1.0], unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(0, 0), binsz=0.1, width=(1, 1), frame="galactic", axes=[energy_axis]
    )
    mDM = 1 * u.TeV
    box_flux = BoxPrimaryFlux(mDM=mDM, mPhi=[100] * u.GeV)
    spectral_model = DarkMatterAnnihilationSpectralModel(
        mDM=mDM,
        channel="b",
        factor=3.41e19 * u.Unit("GeV2 cm-5"),
        primary_flux=box_flux,
    )
    sky_model = SkyModel(
        spectral_model=spectral_model,
        spatial_model=PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic"),
        name="dm_box",
    )
    dataset = MapDataset.create(geom, name="test_box")
    dataset.models = [sky_model]

    assert np.all(np.isfinite(dataset.npred().data))

    # box is non-zero around the boosted photon energy
    E_phi = (mDM + (100 * u.GeV) ** 2 / (4 * mDM)).to("TeV").value
    flux = spectral_model([E_phi * 0.99, E_phi, E_phi * 1.01] * u.TeV)
    assert np.all(np.isfinite(flux.value))
    assert np.any(flux.value > 0)
