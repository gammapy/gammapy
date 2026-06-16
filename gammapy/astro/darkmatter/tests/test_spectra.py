# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from numpy.testing import assert_allclose

from gammapy.astro.darkmatter import (
    BoxPrimaryFlux,
    ContinuumPrimaryFlux,
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    MonochromaticPrimaryFlux,
    VIBPrimaryFlux,
)
from gammapy.modeling.models import Models, SkyModel
from gammapy.utils.testing import assert_quantity_allclose, requires_data

# ---------------------------------------------------------------------------
# ContinuumPrimaryFlux
# ---------------------------------------------------------------------------


@requires_data()
def test_continuum_primary_flux():
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = ContinuumPrimaryFlux(channel="W", mDM=1 * u.TeV)
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
            ContinuumPrimaryFlux(
                channel="aZ",
                mDM=mass * u.TeV,
                source=source,
                mapping_dict={"mDM": "mDM"},
            )
        return
    primflux = ContinuumPrimaryFlux(channel="W", mDM=mass * u.TeV, source=source)
    actual = primflux(500 * u.GeV)
    assert_quantity_allclose(actual, expected_flux / u.GeV, rtol=1e-5)


@requires_data()
def test_primary_flux_cosmixs():
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = ContinuumPrimaryFlux(
        channel="W", mDM=1 * u.TeV, source="cosmixs", mapping_dict={"mDM": "mDM"}
    )
    actual = primflux(500 * u.GeV)
    desired = 0.00013085 / u.GeV
    assert_quantity_allclose(actual, desired, rtol=1e-4)

    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="q", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="V->e", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="V->mu", mDM=1 * u.TeV, source="cosmixs")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="V->tau", mDM=1 * u.TeV, source="cosmixs")

    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="d", mDM=1 * u.TeV, source="pppc4")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="u", mDM=1 * u.TeV, source="pppc4")
    with pytest.raises(ValueError):
        ContinuumPrimaryFlux(channel="s", mDM=1 * u.TeV, source="pppc4")


@requires_data()
def test_continuum_to_from_dict_roundtrip():
    flux = ContinuumPrimaryFlux(channel="W", mDM=1 * u.TeV)
    data = flux.to_dict()
    new_flux = ContinuumPrimaryFlux.from_dict(data)

    assert_quantity_allclose(new_flux.mDM, flux.mDM)
    assert new_flux.channel == flux.channel
    assert new_flux.source == flux.source


def test_custom_source_file_empty(tmp_path):
    """An empty custom source file raises ValueError in the source setter."""
    empty_file = tmp_path / "empty_spectra.dat"
    empty_file.touch()

    with pytest.raises(ValueError, match="Source file is empty"):
        ContinuumPrimaryFlux(mDM=5 * u.TeV, channel="b", source=str(empty_file))


def test_custom_source_file_bad_extension(tmp_path):
    """A custom file with an unsupported extension raises KeyError."""
    bad_file = tmp_path / "spectra.dl2"
    bad_file.write_text("mDM Log[10,x] b\n1000 -3 1e-5\n")

    with pytest.raises(KeyError, match="Source file extension"):
        ContinuumPrimaryFlux(mDM=5 * u.TeV, channel="b", source=str(bad_file))


def test_custom_source_invalid_path():
    """A nonexistent path that is not pppc4/cosmixs raises ValueError."""
    with pytest.raises(ValueError, match="Invalid source"):
        ContinuumPrimaryFlux(
            mDM=5 * u.TeV, channel="b", source="/nonexistent/path.ecsv"
        )


def test_source_non_string_raises_typeerror():
    with pytest.raises(TypeError, match="source must be a string"):
        ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b", source=123)


@requires_data()
def test_source_none_warns_and_defaults_to_pppc4():
    with pytest.warns(UserWarning, match="PPPC4 will be used by default"):
        flux = ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b", source=None)
    assert flux.source == "pppc4"


def test_dm_spectral_model_custom_io(tmp_path):
    """Custom ContinuumPrimaryFlux source survives YAML serialization via the
    spectral model's primary_flux. Note: mapping_dict is NOT serialized by
    ContinuumPrimaryFlux.to_dict, so it is lost on round-trip."""
    custom_file = tmp_path / "custom_spectra.ecsv"

    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0, -3.0, -2.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16] / u.GeV,
        }
    )
    t.write(custom_file, format="ascii.ecsv")

    mapping = {"mDM": "mDM", "Log[10,x]": "Log[10,x]", "b": "b"}

    custom_flux = ContinuumPrimaryFlux(
        mDM=500 * u.GeV, channel="b", source=str(custom_file), mapping_dict=mapping
    )
    assert custom_flux.mapping_dict == mapping

    model = DarkMatterAnnihilationSpectralModel(
        mDM=500 * u.GeV,
        channel="b",
        factor=3.41e19 * u.Unit("GeV2 cm-5"),
        primary_flux=custom_flux,
    )

    sky_model = SkyModel(spectral_model=model, name="skymodel_custom")
    models = Models([sky_model])

    filename = tmp_path / "model_custom.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)
    loaded_model = new_models[0].spectral_model

    assert loaded_model.primary_flux.source == str(custom_file)
    # mapping_dict is not serialized by ContinuumPrimaryFlux.to_dict
    assert loaded_model.primary_flux.mapping_dict is None


def test_dm_annihilation_custom_errors(tmp_path):
    """mapping_dict / channel validation errors arise from ContinuumPrimaryFlux,
    not from the spectral model."""
    file_path = tmp_path / "test_dm.ecsv"
    t = Table()
    t["mDM"] = [1000, 5000] * u.GeV
    t["Log[10,x]"] = [-3, -2]
    t["bbar"] = [1e-5, 1e-4]
    t.write(file_path, format="ascii.ecsv")

    mass = 5 * u.TeV
    with pytest.raises(TypeError, match="mapping_dict must be a dictionary"):
        ContinuumPrimaryFlux(
            mDM=mass,
            channel="b",
            source=str(file_path),
            mapping_dict=["not", "a", "dict"],
        )

    incomplete_mapping = {"Log[10,x]": "Log[10,x]"}
    with pytest.raises(KeyError, match="Mandatory column"):
        ContinuumPrimaryFlux(
            mDM=mass,
            channel="b",
            source=str(file_path),
            mapping_dict=incomplete_mapping,
        )

    wrong_mapping = {"mDM": "mDM", "Log[10,x]": "Log[10,x]", "wrong_col": "tau"}
    with pytest.raises(ValueError, match="is not present in the provided mapping_dict"):
        ContinuumPrimaryFlux(
            mDM=mass, channel="b", source=str(file_path), mapping_dict=wrong_mapping
        )


def test_custom_source_file_without_mapping_and_missing_channel(tmp_path):
    """Custom files without mapping_dict and missing channels."""
    custom_file = tmp_path / "custom_spectra_nomap.ecsv"

    t = Table(
        {
            "mDM": [500.0, 500.0, 1000.0, 1000.0] * u.GeV,
            "Log[10,x]": [-3.0, -2.0, -3.0, -2.0],
            "b": [1e-15, 1e-16, 1e-15, 1e-16],
        }
    )
    t.write(custom_file, format="ascii.ecsv")

    flux = ContinuumPrimaryFlux(mDM=500 * u.GeV, channel="b", source=str(custom_file))
    assert flux.channel == "b"

    with pytest.raises(ValueError, match="is not present in the provided source file"):
        ContinuumPrimaryFlux(mDM=500 * u.GeV, channel="eL", source=str(custom_file))


def test_missing_data_file(monkeypatch):
    """FileNotFoundError is raised if GAMMAPY_DATA points nowhere."""
    monkeypatch.setenv("GAMMAPY_DATA", "/fake/path/to/nowhere")
    with pytest.raises(FileNotFoundError, match="File not found"):
        ContinuumPrimaryFlux(mDM=1 * u.TeV, channel="b")


@requires_data()
def test_mDM_out_of_bounds():
    """ValueError is raised if the mass is out of the interpolation table bounds."""
    with pytest.raises(ValueError, match="is out of bounds"):
        ContinuumPrimaryFlux(mDM=1 * u.eV, channel="b")


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


def test_box_invalid_mphi_length():
    with pytest.raises(ValueError, match="exactly 1 or 2 values"):
        BoxPrimaryFlux(mDM=1 * u.TeV, mPhi=[10, 20, 30] * u.GeV)


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


# ---------------------------------------------------------------------------
# _DarkMatterMassValidator / mDM validation
# ---------------------------------------------------------------------------


def test_mDM_wrong_units_raises():
    with pytest.raises(u.UnitConversionError, match="mDM must have energy units"):
        VIBPrimaryFlux(mDM=1 * u.m)


def test_mDM_zero_or_negative_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        VIBPrimaryFlux(mDM=0 * u.GeV)


# ---------------------------------------------------------------------------
# _AstrophysicalFactorValidator / _RedshiftValidator
# ---------------------------------------------------------------------------


def test_factor_must_be_positive():
    with pytest.raises(
        ValueError, match="astrophysical factor must be strictly positive"
    ):
        DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", factor=-1)


def test_redshift_must_be_non_negative():
    with pytest.raises(ValueError, match="Redshift z must be >= 0"):
        DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", z=-0.1)


def test_redshift_must_be_scalar():
    with pytest.raises(TypeError, match="z must be a dimensionless scalar"):
        DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", z="bad")


# ---------------------------------------------------------------------------
# _PrimaryFluxValidator (mismatch warnings / type checks)
# ---------------------------------------------------------------------------


def test_primary_flux_mass_mismatch_warns():
    """Assigning a primary_flux with mismatched mDM should warn."""
    mDM = 1 * u.TeV
    mono_flux = MonochromaticPrimaryFlux(mDM=2 * u.TeV, n_gamma_photons=2)

    with pytest.warns(UserWarning, match="does not match"):
        DarkMatterAnnihilationSpectralModel(
            mDM=mDM, channel="b", primary_flux=mono_flux
        )


@requires_data()
def test_primary_flux_channel_mismatch_warns():
    """Assigning a ContinuumPrimaryFlux with mismatched channel should warn."""
    mDM = 1 * u.TeV
    cont_flux = ContinuumPrimaryFlux(mDM=mDM, channel="W")

    with pytest.warns(UserWarning, match="does not match"):
        DarkMatterAnnihilationSpectralModel(
            mDM=mDM, channel="b", primary_flux=cont_flux
        )


def test_primary_flux_invalid_type():
    with pytest.raises(TypeError, match="primary_flux must be one of"):
        DarkMatterAnnihilationSpectralModel(
            mDM=1 * u.TeV, channel="b", primary_flux="not_a_flux_model"
        )


def test_decay_expected_primary_flux_mass_is_half():
    """Decay model should expect primary_flux.mDM ~ mDM/2 without warning."""
    import warnings

    mDM = 2 * u.TeV
    mono_flux = MonochromaticPrimaryFlux(mDM=1 * u.TeV, n_gamma_photons=2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = DarkMatterDecaySpectralModel(
            mDM=mDM, channel="b", primary_flux=mono_flux
        )

    assert_quantity_allclose(model.primary_flux.mDM, mDM / 2)


def warnings_should_not_warn(category):
    """Context manager asserting no warning of the given category is raised."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        with pytest.warns(None) as record:
            yield
        for w in record:
            assert not issubclass(w.category, category), (
                f"Unexpected warning: {w.message}"
            )

    return _cm()


# ---------------------------------------------------------------------------
# k parameter (DarkMatterAnnihilationSpectralModel)
# ---------------------------------------------------------------------------


def test_invalid_k_value():
    with pytest.raises(ValueError, match="k must be 2 .Majorana. or 4 .Dirac."):
        DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", k=3)


@requires_data()
@pytest.mark.parametrize("k", [2, 4])
def test_k_value_roundtrip(k):
    model = DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b", k=k)
    data = model.to_dict()
    new_model = DarkMatterAnnihilationSpectralModel.from_dict(data)
    assert new_model.k == k


# ---------------------------------------------------------------------------
# Full spectral models (annihilation / decay) with default ContinuumPrimaryFlux
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_class, factor_unit, expected_flux, expected_dnde, source",
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
            0.0480497420860496,
            0.000231,
            "pppc4",
        ),
        (
            DarkMatterAnnihilationSpectralModel,
            "GeV2 cm-5",
            6.03197683e-14,
            3.52065879e-16,
            "cosmixs",
        ),
        (
            DarkMatterDecaySpectralModel,
            "GeV2 cm-5",
            0.04676,
            0.00027292,
            "cosmixs",
        ),
    ],
)
@requires_data()
def test_dm_spectral_model(
    tmp_path, factor_unit, model_class, expected_flux, expected_dnde, source
):
    channel = "b"
    mass = 5 * u.TeV
    factor = 3.41e19 * u.Unit(factor_unit)
    energy_min = 0.01 * u.TeV
    energy_max = 10 * u.TeV

    pf = ContinuumPrimaryFlux(mass, channel, source=source)
    model = model_class(mDM=mass, channel=channel, factor=factor, primary_flux=pf)

    flux = model.integral(energy_min=energy_min, energy_max=energy_max).to("cm-2 s-1")

    if model_class is DarkMatterDecaySpectralModel:
        dnde = model.evaluate(
            energy=1 * u.TeV, scale=1, lifetime=model.lifetime.quantity
        ).to("cm-2 s-1 TeV-1")
    else:
        dnde = model.evaluate(energy=1 * u.TeV, scale=1).to("cm-2 s-1 TeV-1")

    sky_model = SkyModel(spectral_model=model, name="skymodel")
    models = Models([sky_model])
    filename = tmp_path / "model.yaml"
    models.write(filename, overwrite=True)
    new_models = Models.read(filename)

    assert_quantity_allclose(flux.value, expected_flux, rtol=1e-2)
    assert_quantity_allclose(dnde.value, expected_dnde, rtol=1e-2)

    loaded = new_models[0].spectral_model
    assert loaded.channel == model.channel
    assert loaded.z == model.z
    assert_allclose(loaded.factor.value, model.factor.value, rtol=1e-2)
    assert_quantity_allclose(loaded.mDM, model.mDM, rtol=1e-2)


@requires_data()
def test_dm_annihilation_to_dict_structure():
    """to_dict output structure matches what from_dict expects (regression
    for a previously copy-pasted from_dict docstring on to_dict)."""
    model = DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()

    assert data["spectral"]["channel"] == "b"
    assert data["spectral"]["k"] == 2
    assert "primary_flux" in data["spectral"]
    assert data["spectral"]["primary_flux"]["type"] == "ContinuumPrimaryFlux"

    new_model = DarkMatterAnnihilationSpectralModel.from_dict(data)
    assert new_model.channel == model.channel
    assert new_model.k == model.k


@requires_data()
def test_dm_decay_default_lifetime():
    model = DarkMatterDecaySpectralModel(mDM=1 * u.TeV, channel="b")
    assert_quantity_allclose(model.lifetime.quantity, 4.3e17 * u.s)


@requires_data()
def test_dm_decay_to_from_dict_roundtrip_lifetime():
    model = DarkMatterDecaySpectralModel(
        mDM=1 * u.TeV, channel="b", lifetime=1e20 * u.s
    )
    data = model.to_dict()
    new_model = DarkMatterDecaySpectralModel.from_dict(data)
    assert_quantity_allclose(new_model.lifetime.quantity, model.lifetime.quantity)


@requires_data()
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


@requires_data()
def test_unknown_primary_flux_type_in_from_dict():
    model = DarkMatterAnnihilationSpectralModel(mDM=1 * u.TeV, channel="b")
    data = model.to_dict()
    data["spectral"]["primary_flux"]["type"] = "NotARealFluxType"

    with pytest.raises(ValueError, match="Unknown primary_flux type"):
        DarkMatterAnnihilationSpectralModel.from_dict(data)
