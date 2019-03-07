# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import astropy.units as u
from ...maps import WcsNDMap
from ...image.models import SkyPointSource
from ...cube.models import SkyModel
from ...cube.fit import MapDataset
from ...spectrum.models import AbsorbedSpectralModel
from ...utils.fitting import Fit
from .spectra import DMAnnihilation
from scipy.optimize import brentq
from scipy.interpolate import interp1d
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
    masses. For each fit, the value of the scale parameter that makes :math:`\Delta TS > 2.71` is
    multiplied by the thermal relic cross section, and subsequently taken as the estimated value of
    :math:`\sigma\nu`.

    Parameters
    ----------
    dataset : `~gammapy.cube.fit.MapDataset`
        Simulated dark matter annihilation dataset
    masses : list of `~astropy.units.Quantity`
        List of particle masses where the values of :math:`\sigma\nu` will be calculated
    channels : list of strings allowed in `~gammapy.astro.darkmatter.PrimaryFlux`
        List of channels where the values of :math:`\sigma\nu` will be calculated
    jfact : `~astropy.units.Quantity` (optional)
        Integrated J-Factor needed when `~gammapy.image.models.SkyPointSource` spatial model is used, default value 1
    absorption_model : `~gammapy.spectrum.models.Absorption` (optional)
        Absorption model, default is None
    z: float (optional)
        Redshift value, default value 0
    k: int (optional)
        Type of dark matter particle (k:2 Majorana, k:4 Dirac), default value 2
    xsection: `~astropy.units.Quantity` (optional)
        Thermally averaged annihilation cross-section, default value declared in `~gammapy.astro.darkmatter.DMAnnihilation`

    Examples
    --------
    This is how you may run the `SigmaVEstimator`::

        import logging
        logging.basicConfig()
        logging.getLogger("gammapy.astro.darkmatter.utils").setLevel("INFO")

        jfact = 3.41e19 * u.Unit("GeV2 cm-5")
        channels = ["b", "t", "Z"]
        masses = [70, 200, 500, 5000, 10000, 50000, 100000]*u.GeV
        estimator = SigmaVEstimator(simulated_dataset, masses, channels, jfact=JFAC)
        result = estimator.run()
    """

    def __init__(self, dataset, masses, channels, jfact=1, absorption_model=None, z=0, k=2, xsection=None):

        self.dataset = dataset
        self.masses = masses
        self.channels = channels
        self.jfact = jfact
        self.absorption_model = absorption_model
        self.z = z
        self.k = k

        if not xsection:
            xsection = DMAnnihilation.THERMAL_RELIC_CROSS_SECTION
        self.xsection = xsection

        self._spatial_model = dataset.model.spatial_model
        self._geom = dataset.counts.geom
        self._exposure = dataset.exposure
        self._background_model = dataset.background_model
        self._psf = dataset.psf
        self._edisp = dataset.edisp

        self._counts_map = WcsNDMap(self._geom, np.random.poisson(dataset.npred().data))


    def run(self, optimize_opts=None, covariance_opts=None):
        """Run the SigmaVEstimator for all channels and masses.

        Parameters
        ----------
        optimize_opts : dict
            Options passed to `Fit.optimize`.
        covariance_opts : dict
            Options passed to `Fit.covariance`.

        Returns
        -------
        result : dict
            Dict with results for each channel.
        """

        result = {}
        for ch in self.channels:
            result[ch] = {}
            log.info("Channel: {}".format(ch))
            for mass in self.masses:
                log.info("Mass: {}".format(mass))
                DMAnnihilation.THERMAL_RELIC_CROSS_SECTION = self.xsection
                spectral_model = DMAnnihilation(
                    mass=mass,
                    channel=ch,
                    scale=1,
                    jfactor=self.jfact,
                    z = self.z,
                    k = self.k
                )
                if self.absorption_model:
                    spectral_model = AbsorbedSpectralModel(spectral_model, self.absorption_model, self.z)
                spatial_model = self._spatial_model
                flux_model = SkyModel(
                    spatial_model=spatial_model,
                    spectral_model=spectral_model
                )
                if isinstance(self._spatial_model, SkyPointSource):
                    flux_model.parameters['lat_0'].frozen = True
                    flux_model.parameters['lon_0'].frozen = True

                dataset_loop = MapDataset(
                    model=flux_model,
                    counts=self._counts_map,
                    exposure=self._exposure,
                    background_model=self._background_model,
                    psf=self._psf,
                    edisp=self._edisp
                )
                try:
                    fit = Fit(dataset_loop)
                    fit.datasets.parameters.apply_autoscale = False
                    fit_result = fit.run(optimize_opts, covariance_opts)

                    profile = fit.likelihood_profile(parameter="scale", bounds=5, nvalues=50)
                    xvals = profile["values"]
                    yvals = profile["likelihood"] - fit_result.total_stat - 2.71
                    scale_min = fit_result.parameters["scale"].value
                    scale_max = max(xvals)

                    scale_found = brentq(interp1d(xvals, yvals, kind='cubic'), scale_min, scale_max, maxiter=100, rtol=1e-5)
                    sigma_v = scale_found * self.xsection

                except Exception as ex:
                    sigma_v = None
                    log.error(ex)

                result[ch][mass] = sigma_v
                log.info("Sigma v: {}".format(sigma_v))

        return result

