# End-to-end pipeline test for the DM analysis module.
# Purpose: verify that a complete DM analysis
# (dataset -> model -> fit -> profile likelihood -> upper limit / interval)
# NOT to validate precise scientific results.

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion

from gammapy.data import Observation, FixedPointingInfo
from gammapy.datasets import MapDataset
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    FoVBackgroundModel,
    Models,
    SkyModel,
    PointSpatialModel,
)
from gammapy.astro.darkmatter import (
    DarkMatterDecaySpectralModel,
    DarkMatterAnnihilationSpectralModel,
    profiles,
    JFactory,
)
from gammapy.utils.testing import requires_data, assert_quantity_allclose


IRF_PATH = "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"

TARGET_POS = SkyCoord(ra=260.05167 * u.deg, dec=57.915 * u.deg, frame="icrs")
TARGET_DIST = 76 * u.kpc


def _build_dm_dataset(channel_type, scale, livetime, mass=10 * u.TeV, dm_channel="b"):
    """MapDataset with a DM signal (decay or annihilation) of a given injected `scale`, at a given `livetime`."""
    energy_edges = np.logspace(-1, 2, 8)
    energy_axis = MapAxis.from_edges(
        energy_edges, unit="TeV", name="energy", interp="log"
    )
    energy_axis_true = MapAxis.from_edges(
        energy_edges, unit="TeV", name="energy_true", interp="log"
    )

    geom = WcsGeom.create(
        binsz=0.1, skydir=TARGET_POS, width=2.0, frame="icrs", axes=[energy_axis]
    )

    # Spatial model
    spatial_model = PointSpatialModel(
        lon_0=TARGET_POS.ra, lat_0=TARGET_POS.dec, frame="icrs"
    )

    # Astrophysical factor computation
    r_s = 0.91 * u.kpc
    rho_s = 1.3e7 * (u.M_sun / u.kpc**3)
    rho_s_GeV = rho_s.to(u.GeV / u.cm**3, equivalencies=u.mass_energy())
    density_profile = profiles.EinastoProfile(r_s=r_s, rho_s=rho_s_GeV)

    is_annihilation = channel_type == "annihilation"
    jfactory = JFactory(
        geom=geom,
        profile=density_profile,
        distance=TARGET_DIST,
        annihilation=is_annihilation,
    )
    factor_map = jfactory.compute_jfactor()
    sky_reg = CircleSkyRegion(center=TARGET_POS, radius=0.1 * u.deg)
    pix_reg = sky_reg.to_pixel(wcs=geom.wcs)
    total_factor = pix_reg.to_mask().multiply(factor_map).sum()

    dm_model_name = f"dm-signal-{channel_type}"
    dataset_name = f"dm-dataset-{channel_type}"

    # Spectral model
    if is_annihilation:
        spectral_model = DarkMatterAnnihilationSpectralModel(
            mass=mass, channel=dm_channel, jfactor=total_factor, scale=scale
        )
    else:
        spectral_model = DarkMatterDecaySpectralModel(
            mass=mass, channel=dm_channel, jfactor=total_factor, scale=scale
        )

    # Sky model
    sky_model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name=dm_model_name
    )

    # Background model
    bkg_model = FoVBackgroundModel(dataset_name=dataset_name)
    bkg_model_name = bkg_model.name

    irfs = load_irf_dict_from_file(IRF_PATH)
    obs = Observation.create(
        pointing=FixedPointingInfo(fixed_icrs=TARGET_POS), livetime=livetime, irfs=irfs
    )

    empty = MapDataset.create(
        geom=geom, name=dataset_name, energy_axis_true=energy_axis_true
    )
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    safe_mask_maker = SafeMaskMaker(methods=["offset-max"], offset_max=2.5 * u.deg)

    # Monte Carlo simulation
    dataset = maker.run(empty, obs)
    dataset = safe_mask_maker.run(dataset, obs)
    dataset.models = Models([sky_model, bkg_model])
    dataset.fake(random_state=1)

    return dataset, dm_model_name, bkg_model_name


# Case scenarios
SCENARIOS = [
    pytest.param(
        {
            "channel_type": "decay",
            "scale": 0.0,
            "livetime": 5 * u.h,
            "expect_detection": False,
        },
        id="decay-no-signal",
    ),
    pytest.param(
        {
            "channel_type": "decay",
            "scale": 1e-7,
            "livetime": 20 * u.h,
            "expect_detection": True,
        },
        id="decay-signal-detected",
    ),
    pytest.param(
        {
            "channel_type": "annihilation",
            "scale": 0.0,
            "livetime": 5 * u.h,
            "expect_detection": False,
        },
        id="annihilation-no-signal",
    ),
    pytest.param(
        {
            "channel_type": "annihilation",
            "scale": 1000,
            "livetime": 100 * u.h,
            "expect_detection": True,
        },
        id="annihilation-signal-detected",
    ),
]


@pytest.fixture(params=SCENARIOS)
def dm_scenario(request: pytest.FixtureRequest):
    cfg = request.param
    dataset, dm_model_name, bkg_model_name = _build_dm_dataset(
        channel_type=cfg["channel_type"], scale=cfg["scale"], livetime=cfg["livetime"]
    )
    return dataset, dm_model_name, bkg_model_name, cfg


@requires_data()
def test_dm_analysis_pipeline(dm_scenario):
    """
    End-to-end DM analysis pipeline test, covering both decay and
    annihilation, and both statistical regimes (no signal / signal
    detected), via parametrization.

    For every scenario, checks:
    1. Fit converges.
    2. Predicted counts are finite and non-negative everywhere.
    3. The pipeline reports the statistically correct quantity for the
       given TS regime:
         - no signal injected  -> best fit near scale=0, one-sided 95% CL
           upper limit is computable (Delta TS = 2.71 reached).
         - signal injected     -> best fit clearly nonzero, two-sided
           95% CL interval is computable on both branches around the
           minimum (Delta TS = 3.84 reached on each side).

    """

    dataset, dm_model_name, bkg_model_name, cfg = dm_scenario

    dm_model = dataset.models[dm_model_name]
    bkg_model = dataset.models[bkg_model_name]

    dm_model.spatial_model.lon_0.frozen = True
    dm_model.spatial_model.lat_0.frozen = True

    fit = Fit()

    # H0: background-only
    dm_model.spectral_model.scale.value = 0
    dm_model.spectral_model.scale.min = 0
    dm_model.spectral_model.scale.frozen = True
    bkg_model.parameters["norm"].frozen = False
    bkg_model.parameters["tilt"].frozen = False

    result_h0 = fit.run(datasets=[dataset])
    assert result_h0.success, (
        f"H0 (background-only) fit did not converge for scenario: {cfg}"
    )
    stat_h0 = dataset.stat_sum()

    # H1: full fit
    dm_model.spectral_model.scale.value = max(cfg["scale"], 1e-7)
    dm_model.spectral_model.scale.frozen = False

    result_h1 = fit.run(datasets=[dataset])
    assert result_h1.success, (
        f"H1 (signal+background) fit did not converge for scenario: {cfg}"
    )
    stat_h1 = dataset.stat_sum()

    # Predicted counts are sane
    npred = dataset.npred().data
    assert np.all(np.isfinite(npred)), "npred contains non-finite values"
    assert np.all(npred >= 0), "npred contains negative values"

    # TS
    ts = stat_h0 - stat_h1

    ts_threshold = 25  # 5 sigma, Wilks' theorem, one degree of freedom

    if not cfg["expect_detection"]:
        assert ts < ts_threshold, (
            f"[{cfg}] TS = {ts:.2f} suggests a significant excess "
            f"(> {ts_threshold}) in a dataset with no injected signal -- "
            "possible spurious detection or bug."
        )
    else:
        assert ts > ts_threshold, (
            f"[{cfg}] TS = {ts:.2f} is not significant (< {ts_threshold}) "
            "for a dataset with a clearly injected signal -- the fit may "
            "have failed to detect it."
        )

    # Computing the profile scan to actually derive the limit/interval
    scale_par = dm_model.spectral_model.scale
    if cfg["scale"] > 0:
        scale_par.scan_values = np.linspace(cfg["scale"] * 0.1, cfg["scale"] * 5.0, 200)
    else:
        scale_max = 1e4 if cfg["channel_type"] == "annihilation" else 1e-2
        scale_par.scan_values = np.logspace(-10, np.log10(scale_max), 50)

    profile = fit.stat_profile(datasets=dataset, parameter=scale_par, reoptimize=True)
    scale_scan = profile[f"{dm_model_name}.spectral.scale_scan"]
    delta_ts = profile["stat_scan"] - profile["stat_scan"].min()
    idx_min = np.argmin(delta_ts)

    if not cfg["expect_detection"]:
        # one-sided branch: extract the 95% CL upper limit
        assert delta_ts.max() > 2.71, (
            f"[{cfg}] Could not compute a one-sided 95% CL upper limit -- "
            "profile scan range or fit may be broken."
        )

        # Extract the actual upper limit value via interpolation
        scale_ul = np.interp(2.71, delta_ts[idx_min:], scale_scan[idx_min:])

        assert scale_ul > 0, (
            f"[{cfg}] Extracted upper limit is not positive: {scale_ul}"
        )
        assert np.isfinite(scale_ul), (
            f"[{cfg}] Extracted upper limit is not finite: {scale_ul}"
        )

        if cfg["channel_type"] == "decay":
            expected_ul = 1.188e-08
        else:
            expected_ul = 8.595e02

        assert_quantity_allclose(scale_ul, expected_ul, rtol=1e-3)

    else:
        # two-sided branch: extract the 95% CL interval on both sides
        lo_branch = delta_ts[: idx_min + 1]
        hi_branch = delta_ts[idx_min:]

        assert lo_branch.max() > 3.84 or idx_min == 0, (
            f"[{cfg}] Could not bracket the lower side of the 95% CL "
            "two-sided interval -- scan range may not extend low enough."
        )
        assert hi_branch.max() > 3.84, (
            f"[{cfg}] Could not bracket the upper side of the 95% CL "
            "two-sided interval -- scan range may not extend high enough."
        )

        scale_lo = np.interp(3.84, lo_branch[::-1], scale_scan[: idx_min + 1][::-1])
        scale_hi = np.interp(3.84, hi_branch, scale_scan[idx_min:])

        assert scale_lo > 0 and np.isfinite(scale_lo), (
            f"[{cfg}] Extracted lower bound is not positive/finite: {scale_lo}"
        )
        assert np.isfinite(scale_hi), (
            f"[{cfg}] Extracted upper bound is not finite: {scale_hi}"
        )

        if cfg["channel_type"] == "decay":
            expected_lo = 6.764e-8
            expected_hi = 1.003e-7
        else:
            expected_lo = 768.9
            expected_hi = 1431.896

        assert_quantity_allclose(scale_lo, expected_lo, rtol=1e-3)
        assert_quantity_allclose(scale_hi, expected_hi, rtol=1e-3)

        best_fit_scale = scale_scan[idx_min]
        assert scale_lo < best_fit_scale < scale_hi, (
            f"[{cfg}] Extracted interval [{scale_lo:.2e}, {scale_hi:.2e}] "
            f"does not bracket the best-fit scale ({best_fit_scale:.2e})."
        )
