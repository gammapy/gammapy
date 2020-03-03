# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import astropy.units as u
from gammapy.modeling.models import AbsorbedSpectralModel
from gammapy.modeling import Fit
from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.utils.table import table_from_row_data
from gammapy.astro.darkmatter import DarkMatterAnnihilationSpectralModel
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import logging
import copy

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
    is set by default to 2.71. This process is performed for a given number of runs so to have better statistics.
    Nuisance parameters may be also introduced.

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

        # Define a DMDatasetOnOff dataset
        dataset = DMDatasetOnOff(
            aeff=aeff,
            edisp=edisp,
            models=[flux_model],
            livetime=livetime,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
        )

        # Define channels and masses to run estimator
        channels = ["b", "t", "Z"]
        masses = [70, 200, 500, 5000, 10000, 50000, 100000]*u.GeV

        # Define nuisance parameters and attach them to the dataset
        nuisance = dict(
            j=JFAC,
            jobs=JFAC,
            sigmaj=0.1*JFAC,
            sigmatau=0.01
        )
        dataset.nuisance = nuisance

        # Instantiate the estimator
        estimator = SigmaVEstimator(dataset, masses, channels, background_model=bkg)

        # Run the estimator
        result = estimator.run(10, nuisance=True, stat_profile_opts=dict(bounds=(0, 500), nvalues=100))
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
            background_model
    ):

        self.dataset = dataset
        self.masses = masses
        self.channels = channels
        self.background = background_model
        self.flux_model = dataset.models[0].spectral_model
        if isinstance(dataset.models[0].spectral_model, AbsorbedSpectralModel):
            self.flux_model = dataset.models[0].spectral_model.spectral_model
        self.sigmas = {}
        self.js = {}
        self.result = dict(mean={}, runs={})

    def run(
            self,
            runs,
            nuisance=False,
            stat_profile_opts=None,
            optimize_opts=None,
            covariance_opts=None,
    ):
        """Run the SigmaVEstimator for all channels and masses.

        Parameters
        ----------
        runs: int
            Number of runs where to perform the fitting.
        nuisance: bool
            Flag to perform fitting with nuisance parameters. Default False.
        stat_profile_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.stat_profile`.
        optimize_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.optimize`.
        covariance_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.covariance`.

        Returns
        -------
        result : dict
            result['mean'] provides mean and std values for sigma v vs. mass for each channel.
            result['runs'] provides a table of sigma v vs. mass and likelihood profiles for each run and channel.
        """

        # default options in sv curve
        if stat_profile_opts is None:
            stat_profile_opts = dict(bounds=(-25, 150), nvalues=50)

        # initialize data containers
        for ch in self.channels:
            self.result["mean"][ch] = None
            self.result["runs"][ch] = {}
            self.sigmas[ch] = {}
            self.js[ch] = {}
            for mass in self.masses:
                self.sigmas[ch][mass.value] = {}
                self.js[ch][mass.value] = {}

        okruns = 0
        # loop in runs
        for run in range(runs):
            self.dataset.fake(background_model=self.background)
            valid = self._loops(run, nuisance, stat_profile_opts, optimize_opts, covariance_opts)
            # skip the run and continue with the next on if fails for a mass of a specific channel
            if not valid:
                log.warning(f"Skipping run {run}")
                continue
            else:
                okruns += 1

        # calculate means / std
        if okruns:
            for ch in self.channels:
                table_rows = []
                for mass in self.masses:
                    row = {"mass": mass}
                    listsigmas = [val for key, val in self.sigmas[ch][mass.value].items()]
                    npsigmas = np.array(listsigmas, dtype=np.float)
                    sigma_mean = np.nanmean(npsigmas)
                    sigma_std = np.nanstd(npsigmas)
                    row["sigma_v"] = sigma_mean * self.XSECTION.unit
                    row["sigma_v_std"] = sigma_std * self.XSECTION.unit
                    listjs = [val for key, val in self.js[ch][mass.value].items()]
                    if listjs.count(None) != len(listjs):
                        npjs = np.array(listjs, dtype=np.float)
                        js_mean = np.nanmean(npjs)
                        js_std = np.nanstd(npjs)
                        row["jfactor"] = js_mean * self.dataset.models.parameters["jfactor"].unit
                        row["jfactor_std"] = js_std * self.dataset.models.parameters["jfactor"].unit
                    else:
                        row["jfactor"] = None
                        row["jfactor_std"] = None
                    table_rows.append(row)
                table = table_from_row_data(rows=table_rows)
                self.result["mean"][ch] = table
        log.info(f"Number of good runs: {okruns}")
        return self.result

    def _loops(self, run, nuisance, stat_profile_opts, optimize_opts, covariance_opts):
        """Loop in channels and masses."""

        for ch in self.channels:
            table_rows = []
            for mass in self.masses:

                # modify and set flux model
                dataset_loop = self._set_model_dataset(ch, mass)

                # build profile from fitting
                j_best, sv_best, likemin, statprofile = self._fit_dataset(
                    dataset_loop,
                    nuisance,
                    run,
                    ch,
                    mass,
                    stat_profile_opts=stat_profile_opts,
                    optimize_opts=optimize_opts,
                    covariance_opts=covariance_opts,
                )

                # calculate results from a profile
                fit_result = self._produce_results(j_best, sv_best, likemin, statprofile)

                # not valid run
                if fit_result["sigma_v"] is None:
                    self._make_bad_run_row(run, fit_result)
                    return False

                # build table of results incrementally
                row = {
                    "mass": mass,
                    "sigma_v": fit_result["sigma_v"],
                    "sv_ul": fit_result["sv_ul"],
                    "sv_best": fit_result["sv_best"],
                    "j_best": fit_result["j_best"],
                    "statprofile": fit_result["statprofile"],
                }
                table_rows.append(row)
                self.sigmas[ch][mass.value][run] = row["sigma_v"]
                self.js[ch][mass.value][run] = row["j_best"]
            table = table_from_row_data(rows=table_rows)
            table["sigma_v"].unit = self.XSECTION.unit
            table["j_best"].unit = self.dataset.models.parameters["jfactor"].unit
            self.result["runs"][ch][run] = table
        return True

    def _make_bad_run_row(self, run, fit_result):
        """Add only likelihood profile for a bad run."""

        for ch in self.channels:
            table_rows = []
            for mass in self.masses:
                row = {
                    "mass": mass,
                    "sigma_v": None,
                    "sv_ul": None,
                    "sv_best": None,
                    "j_best": None,
                    "statprofile": fit_result["statprofile"],
                }
                table_rows.append(row)
                self.sigmas[ch][mass.value][run] = None
            table = table_from_row_data(rows=table_rows)
            self.result["runs"][ch][run] = table

    def _set_model_dataset(self, ch, mass):
        """Attach to loop dataset the model to fit."""

        jfactor = self.flux_model.parameters["jfactor"].value * self.flux_model.parameters["jfactor"].unit
        flux_model = DarkMatterAnnihilationSpectralModel(
            mass=mass, channel=ch, sv=1, jfactor=jfactor, z=self.flux_model.z, k=self.flux_model.k
        )
        if isinstance(self.dataset.models[0].spectral_model, AbsorbedSpectralModel):
            flux_model = AbsorbedSpectralModel(
                flux_model, self.dataset.models[0].spectral_model.absorption, self.flux_model.z
            )
        ds = self.dataset.copy()
        ds.models[0].spectral_model = flux_model
        return ds

    @staticmethod
    def _fit_dataset(
            dataset_loop,
            nuisance,
            run,
            ch,
            mass,
            stat_profile_opts=None,
            optimize_opts=None,
            covariance_opts=None,
    ):
        """Fit loop dataset model to fake realization and calculate parameter value for upper limit."""

        log.info(f"----")
        log.info(f"Run: {run}")
        log.info(f"Channel: {ch}")
        log.info(f"Mass: {mass}")

        j_best = None
        stat_profile_opts["parameter"] = "sv"
        dataset_loop.models.parameters["sv"].frozen = False
        if nuisance and dataset_loop.nuisance:
            prior = dataset_loop.nuisance["j"].value
            halfrange = dataset_loop.nuisance["width"] * dataset_loop.nuisance["sigmaj"].value
            dataset_loop.models.parameters["jfactor"].frozen = False
            dataset_loop.models.parameters["jfactor"].min = prior - halfrange
            dataset_loop.models.parameters["jfactor"].max = prior + halfrange
            fit = Fit([dataset_loop])
            fit_result = fit.run("minuit", optimize_opts, covariance_opts)
            sv_best = fit_result.parameters["sv"].value
            j_best = fit_result.parameters["jfactor"].value
            likemin = dataset_loop.stat_sum()
            statprofile = fit.stat_profile(reoptimize=True, **stat_profile_opts)
        else:
            dataset_loop.models.parameters["jfactor"].frozen = True
            fit = Fit([dataset_loop])
            fit_result = fit.run("minuit", optimize_opts, covariance_opts)
            sv_best = fit_result.parameters["sv"].value
            likemin = dataset_loop.stat_sum()
            statprofile = fit.stat_profile(**stat_profile_opts)

        return j_best, sv_best, likemin, statprofile

    def _produce_results(
            self,
            j_best,
            sv_best,
            likemin,
            statprofile
    ):
        """Calculate value of sigma_v from a specific fitting profile."""

        try:
            # check all values are positive
            assert np.max(statprofile["values"]) > 0, "All values found negative"

            # check detection
            likezero = interp1d(
                statprofile["values"],
                statprofile["stat"],
                kind="quadratic",
                fill_value="extrapolate",
            )(0)
            max_like_detection = likezero - likemin
            log.debug(f"ZeroDeltaL: {max_like_detection:.4f} \t| LZero: {likezero:.3f} \t| LMin:  {likemin:.3f}")
            assert max_like_detection <= 25, "Detection found"

            # consider sv value in the physical region
            sv_best_found = sv_best
            if sv_best < 0:
                sv_best = 0
                likemin = likezero

            # consider only right side of the profile
            statprofile = self._make_positive_profile(sv_best, likemin, statprofile)

            # check values in likelihood are bigger than ratio
            likemax = np.max(statprofile['stat'])
            valmax = np.max(statprofile["values"])
            max_like_difference = likemax - likemin
            log.debug(f"MaximumDeltaL: {max_like_difference:.4f} \t| LMax: {likemax:.3f} \t| LMin:  {likemin:.3f}")
            assert max_like_difference > self.RATIO, "Wider range needed in likelihood profile"

            # find the value of the scale parameter `sv` reaching the ratio
            sv_ul = brentq(
                interp1d(
                    statprofile["values"],
                    statprofile["stat"] - likemin - self.RATIO,
                    kind="quadratic"
                ),
                sv_best,
                valmax,
                maxiter=100,
                rtol=1e-5,
            )
            sigma_v = sv_ul * self.XSECTION.value
            log.debug(f"SvBestFound: {sv_best_found:.3f} \t| SvBest: {sv_best:.3f} \t| SvUL: {sv_ul:.3f}")
            if j_best:
                log.debug(f"JBest: {j_best} {self.dataset.models.parameters['jfactor'].unit}")
            log.info(f"Sigma v:{sigma_v} {self.XSECTION.unit}")

        except Exception as ex:
            sigma_v = None
            sv_ul = None
            sv_best = None
            j_best = None
            log.warning(ex)

        res = dict(
            sigma_v=sigma_v,
            sv_ul=sv_ul,
            sv_best=sv_best,
            j_best=j_best,
            statprofile=statprofile,
        )
        return res

    @staticmethod
    def _make_positive_profile(
            sv_best,
            likemin,
            statprofile
    ):
        """Return the right side of the profile."""

        halfprofile = copy.deepcopy(statprofile)
        idx = np.min(np.argwhere(statprofile["values"] > sv_best))
        filtered_x_values = statprofile["values"][statprofile["values"] > sv_best]
        filtered_y_values = statprofile["stat"][idx:]
        halfprofile["values"] = np.concatenate((np.array([sv_best]), filtered_x_values))
        halfprofile["stat"] = np.concatenate((np.array([likemin]), filtered_y_values))
        return halfprofile


class DMDatasetOnOff(SpectrumDatasetOnOff):
    """Dark matter dataset OnOff with nuisance parameters and likelihood."""

    def __init__(self, nuisance=None, **kwargs):
        super().__init__(**kwargs)
        self.nuisance = nuisance

    @property
    def nuisance(self):
        return self._nuisance

    @nuisance.setter
    def nuisance(self, nuisance):
        try:
            if nuisance:
                assert \
                    nuisance["j"].unit == self.models.parameters["jfactor"].unit \
                    and nuisance["jobs"].unit == self.models.parameters["jfactor"].unit \
                    and nuisance["sigmaj"].unit == self.models.parameters["jfactor"].unit, \
                    "Different units in J factor"
                if "width" not in nuisance:
                    nuisance["width"] = 5
            self._nuisance = nuisance
        except Exception as ex:
            log.error(ex)
            self._nuisance = None
            raise ValueError("Nuisance parameters cannot be set")

    def stat_sum(self):
        wstat = super().stat_sum()
        liketotal = wstat
        if self.nuisance:
            liketotal += self.jnuisance() + self.gnuisance()
        return liketotal

    def jnuisance(self):
        jfactor = self.models.parameters["jfactor"].value
        exp_up = (np.log10(jfactor) - np.log10(self.nuisance["jobs"].value)) ** 2
        exp_down = 2 * (np.log(self.nuisance["sigmaj"].value) ** 2)
        up = np.exp(-1 * exp_up / exp_down)
        down = (
                np.log(10)
                * self.nuisance["jobs"].value
                * np.sqrt(2 * np.pi)
                * np.log10(self.nuisance["sigmaj"].value)
        )
        res = up / down
        return -2 * np.log(res)

    def gnuisance(self):
        res = 1 / (np.sqrt(2 * np.pi) * self.nuisance["sigmatau"])
        return -2 * np.log(res)
