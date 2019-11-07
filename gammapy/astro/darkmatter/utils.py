# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import astropy.units as u
from astropy.units.quantity import Quantity
from gammapy.modeling.models import AbsorbedSpectralModel
from gammapy.modeling import Fit
from gammapy.spectrum import SpectrumDatasetOnOff
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
    Nuissance parameters may be also introduced.

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

        # Define a DMDatasetOnOff dataset
        dataset = DMDatasetOnOff(
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

        # Define nuissance parameters and attach them to the dataset
        nuissance = dict(
            j=JFAC,
            jobs=JFAC,
            sigmaj=0.1*JFAC
        )
        dataset.nuissance = nuissance

        # Instantiate the estimator
        estimator = SigmaVEstimator(dataset, masses, channels, background_model=bkg, jfactor=JFAC)

        # Run the estimator
        result = estimator.run(10, nuissance=True, likelihood_profile_opts=dict(bounds=(0, 500), nvalues=100))
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

        # initialization of data containers
        self.sigmas = {}
        self.result = dict(mean={}, runs={})
        for ch in self.channels:
            self.result["mean"][ch] = None
            self.result["runs"][ch] = {}
            self.sigmas[ch] = {}
            for mass in self.masses:
                self.sigmas[ch][mass.value] = {}

    def run(
        self,
        runs,
        nuissance=False,
        likelihood_profile_opts=dict(bounds=100, nvalues=50),
        optimize_opts=None,
        covariance_opts=None,
    ):
        """Run the SigmaVEstimator for all channels and masses.

        Parameters
        ----------
        runs: int
            Number of runs where to perform the fitting.
        nuissance: bool
            Flag to perform fitting with nuissance parameters. Default False.
        likelihood_profile_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.likelihood_profile`.
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
        likelihood_profile_opts["parameter"] = "sv"

        for run in range(runs):
            log.info(f"Run: {run}")
            self.dataset.fake(background_model=self.background)

            # loop in channels and masses
            valid = self._loops(run, nuissance, likelihood_profile_opts, optimize_opts, covariance_opts)
            # if the value of sv<=0 or does not reach self.RATIO
            # skip the run and continue with the next one
            if not valid:
                continue

        # calculate means / stds
        for ch in self.channels:
            table_rows = []
            for mass in self.masses:
                row = {}
                listsigmas = [i for i in self.sigmas[ch][mass.value].values()]
                npsigmas = np.array(listsigmas, dtype=np.float)
                sigma_mean = np.nanmean(npsigmas)
                sigma_std = np.nanstd(npsigmas)
                row["mass"] = mass
                row["sigma_v"] = sigma_mean * self.XSECTION.unit
                row["std"] = sigma_std * self.XSECTION.unit
                table_rows.append(row)
            table = table_from_row_data(rows=table_rows)
            self.result["mean"][ch] = table
        return self.result

    def _loops(self, run, nuissance, likelihood_profile_opts, optimize_opts, covariance_opts):
        for ch in self.channels:
            log.info(f"Channel: {ch}")
            table_rows = []
            for mass in self.masses:
                log.info(f"Mass: {mass}")
                dataset_loop = self._set_model_dataset(ch, mass)
                fit_result = self._fit_dataset(
                    dataset_loop,
                    nuissance,
                    run,
                    ch,
                    mass,
                    likelihood_profile_opts=likelihood_profile_opts,
                    optimize_opts=optimize_opts,
                    covariance_opts=covariance_opts,
                )
                if fit_result["sigma_v"] is None:
                    self._make_bad_run_row(run, fit_result)
                    return False

                row = {
                    "mass": mass,
                    "sigma_v": fit_result["sigma_v"].value,
                    "sv_best": fit_result["sv_best"],
                    "sv_ul": fit_result["sv_ul"],
                    "likeprofile": fit_result["likeprofile"],
                }
                table_rows.append(row)
                self.sigmas[ch][mass.value][run] = row["sigma_v"]
                log.info(f"Sigma v:{row['sigma_v']}")

            table = table_from_row_data(rows=table_rows)
            table["sigma_v"].unit = self.XSECTION.unit
            self.result["runs"][ch][run] = table
        return True

    def _make_bad_run_row(self, run, fit_result):
        """Add the likelihood profile for a bad run."""
        for ch in self.channels:
            table_rows = []
            for mass in self.masses:
                row = {
                    "mass": mass,
                    "sigma_v": None,
                    "sv_best": None,
                    "sv_ul": None,
                    "likeprofile": fit_result["likeprofile"],
                }
                table_rows.append(row)
                self.sigmas[ch][mass.value][run] = None
            table = table_from_row_data(rows=table_rows)
            self.result["runs"][ch][run] = table

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
        nuissance,
        run,
        ch,
        mass,
        likelihood_profile_opts=None,
        optimize_opts=None,
        covariance_opts=None,
    ):
        """Fit dataset to model and calculate parameter value for upper limit."""

        if nuissance and dataset_loop.check_nuissance():
            # fit to the realization for each value of j nuissance parameter in a range of sigmaj
            resfits = []
            likes = []
            profiles = []
            js = []
            widthsigma = dataset_loop.nuissance["width"] * dataset_loop.nuissance["sigmaj"].value
            jlo = dataset_loop.nuissance["jobs"].value - widthsigma
            jhi = dataset_loop.nuissance["jobs"].value + widthsigma
            unit = dataset_loop.nuissance["j"].unit

            for ji in np.linspace(jlo, jhi, dataset_loop.nuissance["steps"]):
                dataset_loop.nuissance["j"] = ji * unit
                ifit = Fit(dataset_loop)
                js.append(ji)
                resfits.append(ifit.run())
                ilike = dataset_loop.likelihood()
                likes.append(ilike)
                profiles.append(ifit.likelihood_profile(**likelihood_profile_opts))
                log.debug(f"J: {ji:.2e} \t Min Likelihood: {ilike}")
            # choose likelihood profile giving the minimum value for the likelihood
            likemin = min(likes)
            idx = likes.index(likemin)
            likeprofile = profiles[idx]
            fit_result = resfits[idx]
            log.debug(f"J best: {js[idx]}")
        else:
            fit = Fit(dataset_loop)
            fit_result = fit.run(optimize_opts, covariance_opts)
            likeprofile = fit.likelihood_profile(**likelihood_profile_opts)
            likemin = dataset_loop.likelihood()
        halfprofile = copy.deepcopy(likeprofile)

        # consider sv value in the physical region
        sv_best = fit_result.parameters["sv"].value
        log.debug(f"SvBest found: {sv_best}")
        if sv_best < 0:
            sv_best = 0
            likemin = interp1d(likeprofile["values"], likeprofile["likelihood"], kind="quadratic")(0)

        max_like_detection = 0
        max_like_difference = 0
        if max(halfprofile["values"] > 0):
            idx = np.min(np.argwhere(likeprofile["values"] > 0))
            filtered_x_values = likeprofile["values"][likeprofile["values"] > 0]
            filtered_y_values = likeprofile["likelihood"][idx:]
            halfprofile["values"] = np.concatenate((np.array([sv_best]), filtered_x_values))
            halfprofile["likelihood"] = np.concatenate((np.array([likemin]), filtered_y_values))
            max_like_difference = (np.max(halfprofile["likelihood"]) - likemin - self.RATIO)
            # detection
            likezero = interp1d(
                likeprofile["values"],
                likeprofile["likelihood"],
                kind="quadratic",
                fill_value="extrapolate",
            )(0)
            max_like_detection = likezero - likemin

            log.debug(f"Min Likelihood: {likemin}")
            log.debug(f"SvBest: {sv_best}")
            log.debug(f"SvMax: {np.max(halfprofile['values'])}")
            log.debug(f"DeltaLMax: {max_like_difference:.4f} \t| Max:  {np.max(halfprofile['likelihood'])}")
            log.debug(f"DeltaLZero: {max_like_detection:.4f} \t| Zero: {likezero}")

        try:
            assert (np.max(halfprofile["values"]) > 0), "Values for jfactor found outside the physical region"
            assert max_like_difference > 0, "Wider range needed in likelihood profile"
            assert max_like_detection <= 25, "Detection found"

            # find the value of the scale parameter `sv` reaching self.RATIO
            sv_ul = brentq(
                interp1d(
                    halfprofile["values"],
                    halfprofile["likelihood"] - likemin - self.RATIO,
                    kind="quadratic",
                ),
                sv_best,
                np.max(halfprofile["values"]),
                maxiter=100,
                rtol=1e-5,
            )
            sigma_v = sv_ul * self.XSECTION
        except Exception as ex:
            sigma_v = None
            sv_best = None
            sv_ul = None
            log.warning(f"Skipping Run: {run}")
            log.warning(f"Channel: {ch} - Mass: {mass}")
            log.warning(ex)

        res = dict(
            sigma_v=sigma_v,
            sv_best=sv_best,
            sv_ul=sv_ul,
            likeprofile=likeprofile,
        )
        return res


class DMDatasetOnOff(SpectrumDatasetOnOff):
    """Dark matter dataset OnOff with nuissance parameters and likelihood."""

    def __init__(self, nuissance=None, **kwargs):
        super().__init__(**kwargs)
        if nuissance is None:
            nuissance = {
                "j": None,
                "jobs": None,
                "sigmaj": None,
                "width": None,
                "steps": None,
            }
        self.nuissance = nuissance

    @property
    def nuissance(self):
        return self._nuissance

    @nuissance.setter
    def nuissance(self, nuissance):
        if "width" not in nuissance:
            nuissance["width"] = 5
        if "steps" not in nuissance:
            nuissance["steps"] = 15
        self._nuissance = nuissance

    def check_nuissance(self):
        if not isinstance(self.nuissance["j"], Quantity):
            return False
        if not isinstance(self.nuissance["jobs"], Quantity):
            return False
        if not isinstance(self.nuissance["sigmaj"], Quantity):
            return False
        if not self.nuissance["j"].value:
            return False
        if not self.nuissance["jobs"].value:
            return False
        if not self.nuissance["sigmaj"].value:
            return False
        if not self.nuissance["width"]:
            return False
        if not self.nuissance["steps"]:
            return False
        return True

    def likelihood(self):
        wstat = super().likelihood()
        liketotal = wstat
        if self.check_nuissance():
            liketotal += self.jnuissance()
        return liketotal

    def jnuissance(self):
        exp_up = (np.log10(self.nuissance["j"].value) - np.log10(self.nuissance["jobs"].value)) ** 2
        exp_down = 2 * (np.log(self.nuissance["sigmaj"].value) ** 2)
        up = np.exp(-1 * exp_up / exp_down)
        down = (
            np.log(10)
            * self.nuissance["jobs"].value
            * np.sqrt(2 * np.pi)
            * np.log10(self.nuissance["sigmaj"].value)
        )
        res = up / down
        return -2 * np.log(res)
