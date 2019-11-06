# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import astropy.units as u
from astropy.units import Quantity
from gammapy.modeling.models import AbsorbedSpectralModel
from gammapy.modeling import Fit
from gammapy.utils.table import table_from_row_data
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import logging

__all__ = ["JFactory", "SigmaVEstimator"]

log = logging.getLogger(__name__)


class JFactory:
    """Compute J-Factor maps.

    All J-Factors are computed for annihilation. The assumed dark matter
    profiles will be centered on the center of the map.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry
    profile : `~gammapy.astro.darkmatter.profiles.DMProfile`
        Dark matter profile
    distance : `~astropy.units.Quantity`
        Distance to convert angular scale of the map
    """

    def __init__(self, geom, profile, distance):
        self.geom = geom
        self.profile = profile
        self.distance = distance

    def compute_differential_jfactor(self):
        r"""Compute differential J-Factor.

        .. math::
            \frac{\mathrm d J}{\mathrm d \Omega} =
            \int_{\mathrm{LoS}} \mathrm d r \rho(r)
        """
        # TODO: Needs to be implemented more efficiently
        separation = self.geom.separation(self.geom.center_skydir)
        rmin = separation.rad * self.distance
        rmax = self.distance
        val = [self.profile.integral(_, rmax) for _ in rmin.flatten()]
        jfact = u.Quantity(val).to("GeV2 cm-5").reshape(rmin.shape)
        return jfact / u.steradian

    def compute_jfactor(self):
        r"""Compute astrophysical J-Factor.

        .. math::
            J(\Delta\Omega) =
           \int_{\Delta\Omega} \mathrm d \Omega^{\prime}
           \frac{\mathrm d J}{\mathrm d \Omega^{\prime}}
        """
        diff_jfact = self.compute_differential_jfactor()
        return diff_jfact * self.geom.to_image().solid_angle()


class SigmaVEstimator:
    r"""Estimates :math:`\sigma\nu` for a list of annihilation channels and particle masses.

    To estimate the different values of :math:`\sigma\nu`, a random poisson realization for a given
    annihilation simulated dataset is fitted to a list of `~gammapy.astro.darkmatter.DarkMatterAnnihilationSpectralModel`
    models. These are created within the range of the given lists of annihilation channels and particle
    masses. For each fit, the value of the scale parameter (in the range of the physical region >=0)
    that makes the likelihood ratio :math:`-2\lambda_P = RATIO` is multiplied by the thermal relic cross
    section, and subsequently taken as the estimated value of :math:`\sigma\nu`. The value of :math:`RATIO`
    is set by default to 2.71 and may be modified as an attribute of this `SigmaVEstimator` class.

    Parameters
    ----------
    dataset : `~gammapy.spectrum.dataset.SpectrumDatasetOnOff`
        Simulated dark matter annihilation spectrum OnOff dataset.
    masses : list of `~astropy.units.Quantity`
        List of particle masses where the values of :math:`\sigma\nu` will be calculated.
    channels : list of strings allowed in `~gammapy.astro.darkmatter.PrimaryFlux`
        List of channels where the values of :math:`\sigma\nu` will be calculated.
    background_model: `~gammapy.spectrum.CountsSpectrum`
        BackgroundModel. In the future will be part of the SpectrumDataset Class.
        For the moment, a CountSpectrum.
    jfact : `~astropy.units.Quantity` (optional)
        Integrated J-Factor
        Needed when `~gammapy.image.models.SkyPointSource` spatial model is used.
        Default value 1.

    Examples
    --------
    This is how you may run the `SigmaVEstimator`::

        import logging
        logging.basicConfig()
        logging.getLogger("gammapy.astro.darkmatter.utils").setLevel("INFO")

        # Define annihilation model
        JFAC = 3.41e19 * u.Unit("GeV2 cm-5")
        flux_model = DarkMatterAnnihilationSpectralModel(mass=5000*u.GeV, channel="b", jfactor=JFAC)

        # Define a background model for the off counts as a CountsSpectrum
        bkg = CountsSpectrum(energy_min, energy_max, data=offcounts)

        # Define an empty SpectrumDataSetOnOff dataset
        dataset = SpectrumDatasetOnOff(
            aeff=aeff,
            edisp=edisp,
            model=flux_model,
            livetime=livetime,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
        )

        # Define channels and masses to run estimator
        channels = ["b", "t", "Z"]
        masses = [70, 200, 500, 5000, 10000, 50000, 100000]*u.GeV

        # Define nuissance parameters
        nuissance = dict(
            j=JFAC,
            jobs=JFAC,
            sigmaj=0.1*JFAC
        )

        # Instantiate the estimator
        estimator = SigmaVEstimator(dataset, masses, channels, background_model=bkg, jfactor=JFAC)

        # Run the estimator
        result = estimator.run(10, nuissance=nuissance, likelihood_profile_opts=dict(bounds=(0, 500), nvalues=100))
    """

    RATIO = 2.71
    """Value for the likelihood ratio criteria - set to 2.71 by default."""

    XSECTION = DarkMatterAnnihilationSpectralModel.THERMAL_RELIC_CROSS_SECTION
    """Value for the thermal relic cross section."""

    def __init__(
        self,
        dataset,
        masses,
        channels,
        background_model,
        jfactor=1,
    ):


        self.dataset = dataset
        self.masses = masses
        self.channels = channels
        self.background = background_model
        self.jfactor = jfactor

        dm_params_container = dataset.model
        if isinstance(dataset.model, AbsorbedSpectralModel):
            dm_params_container = dataset.model.spectral_model
        self.z = dm_params_container.z
        self.k = dm_params_container.k


    def run(
        self,
        runs,
        nuissance=None,
        likelihood_profile_opts=dict(bounds=100, nvalues=50),
        optimize_opts=None,
        covariance_opts=None,
    ):
        """Run the SigmaVEstimator for all channels and masses.

        Parameters
        ----------
        runs: int
            Number of runs where to perform the fitting
        nuissance: dict
            Dictionary with nuissance parameters
        likelihood_profile_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.likelihood_profile`.
        optimize_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.optimize`.
        covariance_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.covariance`.

        Returns
        -------
        result : dict
            Nested dict of channels with results in `~astropy.table.Table` objects for each channel.
            result['mean'] provides mean values for sigma v vs. mass.
            result['runs'] provides sigma v vs. mass. and profile likelihood for each channel and run.
        """
        likelihood_profile_opts["parameter"] = "sv"

        # initialization of data containers
        sigmas = {}
        result = dict(mean={}, runs={})
        for ch in self.channels:
            result["mean"][ch] = None
            result["runs"][ch] = []
            sigmas[ch] = {}
            for mass in self.masses:
                sigmas[ch][mass.value] = []
        sigma_unit = ""

        for run in range(runs):
            log.info(f"Run: {run}")
            self.dataset.fake(background_model=self.background)
            for ch in self.channels:
                log.info(f"Channel: {ch}")
                table_rows = []
                for mass in self.masses:
                    log.info(f"Mass: {mass}")
                    row = {}
                    DarkMatterAnnihilationSpectralModel.THERMAL_RELIC_CROSS_SECTION = self.xsection
                    dataset_loop = self._set_model_dataset(ch, mass)
                    fit_result = self._fit_dataset(
                        dataset_loop,
                        ch,
                        run,
                        mass,
                        likelihood_profile_opts=likelihood_profile_opts,
                        optimize_opts=optimize_opts,
                        covariance_opts=covariance_opts,
                    )
                    row["mass"] = mass
                    if isinstance(fit_result["sigma_v"], Quantity):
                        row["sigma_v"] = fit_result["sigma_v"].value
                        sigma_unit = fit_result["sigma_v"].unit
                    else:
                        row["sigma_v"] = fit_result["sigma_v"]
                    row["sv_best"] = fit_result["sv_best"]
                    row["sv_ul"] = fit_result["sv_ul"]
                    row["likeprofile"] = fit_result["likeprofile"]
                    table_rows.append(row)
                    sigmas[ch][mass.value].append(row["sigma_v"])
                    log.info(f"Sigma v:{row['sigma_v']}")
                table = table_from_row_data(rows=table_rows)
                table["sigma_v"].unit = sigma_unit
                result["runs"][ch].append(table)

        # calculate mean results
        for ch in self.channels:
            table_rows = []
            for mass in self.masses:
                row = {}
                npsigmas = np.array(sigmas[ch][mass.value], dtype=np.float)
                sigma_mean = np.nanmean(npsigmas)
                sigma_std = np.nanstd(npsigmas)
                row["mass"] = mass
                row["sigma_v"] = sigma_mean * sigma_unit
                row["std"] = sigma_std * sigma_unit
                table_rows.append(row)
            table = table_from_row_data(rows=table_rows)
            result["mean"][ch] = table
        return result

    def _set_model_dataset(self, ch, mass):
        """Set model to fit in dataset."""
        flux_model = DarkMatterAnnihilationSpectralModel(
            mass=mass, channel=ch, sv=1, jfactor=self.jfactor, z=self.z, k=self.k
        )
        if isinstance(self.dataset.model, AbsorbedSpectralModel):
            flux_model = AbsorbedSpectralModel(
                flux_model, self.dataset.model.absorption, self.z
            )
        ds = self.dataset.copy()
        ds.model = flux_model
        return ds

    def _fit_dataset(
        self,
        dataset_loop,
        ch,
        run,
        mass,
        nuissance=None,
        likelihood_profile_opts=None,
        optimize_opts=None,
        covariance_opts=None,
    ):
        """Fit dataset to model and calculate parameter value for upper limit."""
        fit = Fit(dataset_loop)
        fit_result = fit.run(optimize_opts, covariance_opts)
        likeprofile = fit.likelihood_profile(**likelihood_profile_opts)
        sv_best = fit_result.parameters["sv"].value
        likemin = dataset_loop.likelihood()
        profile = likeprofile

        # consider sv value in the physical region > 0
        if sv_best < 0:
            sv_best = 0
            likemin = interp1d(likeprofile["values"], likeprofile["likelihood"], kind="quadratic")(0)
            idx = np.min(np.argwhere(likeprofile["values"] > 0))
            filtered_x_values = likeprofile["values"][likeprofile["values"] > 0]
            filtered_y_values = likeprofile["likelihood"][idx:]
            profile["values"] = np.concatenate((np.array([0]), filtered_x_values))
            profile["likelihood"] = np.concatenate(
                (np.array([likemin]), filtered_y_values)
            )
        max_like_difference = np.max(profile["likelihood"]) - likemin - self.RATIO

        try:
            assert max_like_difference > 0, "Wider range needed in likelihood profile"
            assert (
                np.max(profile["values"]) > 0
            ), "Values for jfactor found outside the physical region"
            sv_ul = brentq(
                interp1d(
                    profile["values"],
                    profile["likelihood"] - likemin - self.RATIO,
                    kind="quadratic",
                ),
                sv_best,
                np.max(profile["values"]),
                maxiter=100,
                rtol=1e-5,
            )
            sigma_v = sv_ul * self.XSECTION
        except Exception as ex:
            sigma_v = None
            sv_best = None
            sv_ul = None
            log.warning(f"Channel: {ch} - Run: {run} - Mass: {mass}")
            log.warning(ex)

        res = dict(
            sigma_v=sigma_v,
            sv_best=sv_best,
            sv_ul=sv_ul,
            likeprofile=likeprofile,
        )
        return res
