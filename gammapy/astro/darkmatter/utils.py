# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import astropy.units as u
from astropy.units import Quantity
from gammapy.modeling.models import AbsorbedSpectralModel
from gammapy.spectrum import SpectrumDatasetOnOff
from gammapy.modeling import Fit
from gammapy.utils.table import table_from_row_data
from gammapy.astro.darkmatter import DMAnnihilation
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
    annihilation simulated dataset is fitted to a list of `~gammapy.astro.darkmatter.DMAnnihilation`
    models. These are created within the range of the given lists of annihilation channels and particle
    masses. For each fit, the value of the scale parameter (in the range of the physical region >=0)
    that makes the likelihood ratio :math:`-2\lambda_P = RATIO` is multiplied by the thermal relic cross
    section, and subsequently taken as the estimated value of :math:`\sigma\nu`. The value of :math:`RATIO`
    is set by default to 2.71 and may be modified as an attribute of this `SigmaVEstimator` class.

    Parameters
    ----------
    dataset : `~gammapy.spectrum.dataset.SpectrumDatasetOnOff`
        Simulated dark matter annihilation spectrum dataset.
    masses : list of `~astropy.units.Quantity`
        List of particle masses where the values of :math:`\sigma\nu` will be calculated.
    channels : list of strings allowed in `~gammapy.astro.darkmatter.PrimaryFlux`
        List of channels where the values of :math:`\sigma\nu` will be calculated.
    background_model: `~gammapy.spectrum.CountsSpectrum`
        BackgroundModel. In the future will be part of the SpectrumDataset Class.
        For the moment, a CountSpectrum.
    jfact : `~astropy.units.Quantity` (optional)
        Integrated J-Factor needed when `~gammapy.image.models.SkyPointSource` spatial model is used, default value 1.
    absorption_model : `~gammapy.spectrum.models.Absorption` (optional)
        Absorption model, default is None.
    z: float (optional)
        Redshift value, default value 0.
    k: int (optional)
        Type of dark matter particle (k:2 Majorana, k:4 Dirac), default value 2.
    xsection: `~astropy.units.Quantity` (optional)
        Thermally averaged annihilation cross-section, default value declared in `~gammapy.astro.darkmatter.DMAnnihilation`.

    Examples
    --------
    This is how you may run the `SigmaVEstimator`::

        import logging
        logging.basicConfig()
        logging.getLogger("gammapy.astro.darkmatter.utils").setLevel("INFO")

        # Create annihilation model
        JFAC = 3.41e19 * u.Unit("GeV2 cm-5")
        flux_model = DMAnnihilation(mass=5000*u.GeV, channel="b", jfactor=JFAC)

        # Create an empty SpectrumDataSetOnOff dataset
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
        estimator = SigmaVEstimator(dataset, masses, channels, background_model=bkg, jfact=JFAC)
        result = estimator.run(likelihood_profile_opts=dict(bounds=(0, 500), nvalues=100))
    """

    RATIO = 2.71
    """Value for the likelihood ratio criteria - set to 2.71 by default."""

    def __init__(
        self,
        dataset,
        masses,
        channels,
        background_model,
        jfact=1,
        absorption_model=None,
        z=0,
        k=2,
        xsection=None,
    ):

        self.dataset = dataset
        self.masses = masses
        self.channels = channels
        self.background = background_model
        self.jfact = jfact
        self.absorption_model = absorption_model
        self.z = z
        self.k = k

        if not xsection:
            xsection = DMAnnihilation.THERMAL_RELIC_CROSS_SECTION
        self.xsection = xsection

    def run(
        self,
        likelihood_profile_opts=dict(bounds=5, nvalues=50),
        optimize_opts=None,
        covariance_opts=None,
    ):
        """Run the SigmaVEstimator for all channels and masses.

        Parameters
        ----------
        likelihood_profile_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.likelihood_profile`.
        optimize_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.optimize`.
        covariance_opts : dict
            Options passed to `~gammapy.utils.fitting.Fit.covariance`.

        Returns
        -------
        result : dict
            Dict with results as `~astropy.table.Table` objects for each channel.
        """

        self.dataset.fake(background_model=self.background)
        likelihood_profile_opts["parameter"] = "scale"

        result = {}
        sigma_unit = ""
        for ch in self.channels:
            table_rows = []
            log.info(f"Channel: {ch}")
            for mass in self.masses:
                row = {}
                log.info(f"Mass: {mass}")
                DMAnnihilation.THERMAL_RELIC_CROSS_SECTION = self.xsection
                dataset_loop = self._set_model_dataset(ch, mass)
                fit_result = self._fit_dataset(
                    dataset_loop,
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
                row["scale_best"] = fit_result["scale_best"]
                row["scale_ul"] = fit_result["scale_ul"]
                row["likeprofile"] = fit_result["likeprofile"]
                table_rows.append(row)
                log.info(f"Sigma v: {fit_result['sigma_v']}")
            table = table_from_row_data(rows=table_rows)
            table["sigma_v"].unit = sigma_unit
            result[ch] = table
        return result

    def _set_model_dataset(self, ch, mass):
        """Set model to fit in dataset."""
        flux_model = DMAnnihilation(
            mass=mass, channel=ch, scale=1, jfactor=self.jfact, z=self.z, k=self.k
        )
        if self.absorption_model:
            flux_model = AbsorbedSpectralModel(
                flux_model, self.absorption_model, self.z
            )
        ds = self.dataset.copy()
        ds.model = flux_model
        return ds

    def _fit_dataset(
        self,
        dataset_loop,
        likelihood_profile_opts=None,
        optimize_opts=None,
        covariance_opts=None,
    ):
        """Fit dataset to model."""
        try:
            fit = Fit(dataset_loop)
            fit_result = fit.run(optimize_opts, covariance_opts)
            likeprofile = fit.likelihood_profile(**likelihood_profile_opts)
            #
            #
            scale_best = fit_result.parameters["scale"].value
            #
            likemin = dataset_loop.likelihood()
            profile = likeprofile

            # consider scale value in the physical region > 0
            assert (np.max(profile["values"]) > 0), "Values for scale found outside the physical region"
            if scale_best < 0:
                scale_best = 0
                likemin = interp1d(likeprofile["values"], likeprofile["likelihood"], kind="quadratic")(0)
                idx = np.min(np.argwhere(likeprofile["values"] > 0))
                filtered_x_values = likeprofile["values"][likeprofile["values"] > 0]
                filtered_y_values = likeprofile["likelihood"][idx:]
                profile["values"] = np.concatenate((np.array([0]), filtered_x_values))
                profile["likelihood"] = np.concatenate((np.array([likemin]), filtered_y_values))
            scale_ul = brentq(
                interp1d(
                    profile["values"],
                    profile["likelihood"] - likemin - self.RATIO,
                    kind="quadratic",
                ),
                scale_best,
                np.max(profile["values"]),
                maxiter=100,
                rtol=1e-5,
            )
            sigma_v = scale_ul * self.xsection

        except Exception as ex:
            sigma_v = None
            scale_best = None
            scale_ul = None
            likeprofile = None
            log.error(ex)

        res = dict(
            sigma_v=sigma_v,
            scale_best=scale_best,
            scale_ul=scale_ul,
            likeprofile=likeprofile,
        )
        return res
