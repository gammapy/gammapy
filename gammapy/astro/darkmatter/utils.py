# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities to compute J-factor maps."""
import astropy.units as u
from astropy.units import Quantity
from ...maps import WcsNDMap
from ...image.models import SkyPointSource
from ...cube.models import SkyModel
from ...cube.fit import MapDataset
from ...spectrum.models import AbsorbedSpectralModel
from ...utils.fitting import Fit
from ...utils.table import table_from_row_data
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
        Simulated dark matter annihilation dataset.
    masses : list of `~astropy.units.Quantity`
        List of particle masses where the values of :math:`\sigma\nu` will be calculated.
    channels : list of strings allowed in `~gammapy.astro.darkmatter.PrimaryFlux`
        List of channels where the values of :math:`\sigma\nu` will be calculated.
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
        from gammapy.cube.simulate import simulate_dataset
        from gammapy.maps import WcsGeom, MapAxis
        from gammapy.cube.models import SkyModel
        from gammapy.image.models import SkyPointSource
        from gammapy.astro.darkmatter import DMAnnihilation, SigmaVEstimator
        from gammapy.irf import load_cta_irfs
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        import numpy as np

        # Create point source map
        GLON = 96.34 * u.deg
        GLAT = -60.19 * u.deg
        src_pos = SkyCoord(GLON, GLAT, frame="galactic")
        irfs = load_cta_irfs("$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits")
        axis = MapAxis.from_edges(np.logspace(np.log10(0.01), np.log10(100), 30), unit="TeV", name="energy", interp="log")
        geom = WcsGeom.create(skydir=src_pos, binsz=0.02, width=(2, 2), coordsys="GAL", axes=[axis])
        spatial_model = SkyPointSource(lat_0=GLAT, lon_0=GLON)

        # Create annihilation model
        JFAC = 3.41e19 * u.Unit("GeV2 cm-5")
        flux_model = DMAnnihilation(mass=5000*u.GeV, channel="b", jfactor=JFAC, z=5)

        # Combine into sky model
        sky_model = SkyModel(spatial_model=spatial_model, spectral_model=flux_model)
        sim_dataset = simulate_dataset(sky_model, geom=geom, pointing=src_pos, irfs=irfs, livetime=50*u.hour, offset=2*u.deg)

        # Define channels and masses to run estimator
        channels = ["b", "t", "Z"]
        masses = [70, 200, 500, 5000, 10000, 50000, 100000]*u.GeV
        estimator = SigmaVEstimator(sim_dataset, masses, channels, jfact=JFAC)
        result = estimator.run(likelihood_profile_opts=dict(bounds=3, nvalues=25))
    """

    def __init__(
        self,
        dataset,
        masses,
        channels,
        jfact=1,
        absorption_model=None,
        z=0,
        k=2,
        xsection=None,
    ):

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

    def run(
        self, likelihood_profile_opts=None, optimize_opts=None, covariance_opts=None
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

        spatial_model = self.dataset.model.spatial_model
        counts_map = WcsNDMap(
            self.dataset.counts.geom, np.random.poisson(self.dataset.npred().data)
        )

        if likelihood_profile_opts is None:
            likelihood_profile_opts = {"bounds": 5, "nvalues": 50}
        likelihood_profile_opts["parameter"] = "scale"

        result = {}
        sigma_unit = ""
        for ch in self.channels:
            tablerows = []
            log.info("Channel: {}".format(ch))
            for mass in self.masses:
                row = {}
                log.info("Mass: {}".format(mass))
                DMAnnihilation.THERMAL_RELIC_CROSS_SECTION = self.xsection
                spectral_model = DMAnnihilation(
                    mass=mass,
                    channel=ch,
                    scale=1,
                    jfactor=self.jfact,
                    z=self.z,
                    k=self.k,
                )
                if self.absorption_model:
                    spectral_model = AbsorbedSpectralModel(
                        spectral_model, self.absorption_model, self.z
                    )
                flux_model = SkyModel(
                    spatial_model=spatial_model, spectral_model=spectral_model
                )
                if isinstance(spatial_model, SkyPointSource):
                    flux_model.parameters["lat_0"].frozen = True
                    flux_model.parameters["lon_0"].frozen = True

                dataset_loop = MapDataset(
                    model=flux_model,
                    counts=counts_map,
                    exposure=self.dataset.exposure,
                    background_model=self.dataset.background_model,
                    psf=self.dataset.psf,
                    edisp=self.dataset.edisp,
                )
                try:
                    fit = Fit(dataset_loop)
                    fit_result = fit.run(optimize_opts, covariance_opts)
                    likemin = fit_result.total_stat
                    profile = fit.likelihood_profile(**likelihood_profile_opts)
                    xvals = profile["values"]
                    yvals = profile["likelihood"] - likemin - 2.71
                    scale_min = fit_result.parameters["scale"].value
                    scale_max = max(xvals)

                    scale_found = brentq(
                        interp1d(xvals, yvals, kind="cubic"),
                        scale_min,
                        scale_max,
                        maxiter=100,
                        rtol=1e-5,
                    )
                    sigma_v = scale_found * self.xsection

                except Exception as ex:
                    sigma_v = None
                    scale_min = None
                    profile = None
                    likemin = None
                    log.error(ex)

                row["mass"] = mass
                if isinstance(sigma_v, Quantity):
                    row["sigma_v"] = sigma_v.value
                    sigma_unit = sigma_v.unit
                else:
                    row["sigma_v"] = sigma_v
                row["scale"] = scale_min
                row["likeprofile"] = profile
                row["likemin"] = likemin
                tablerows.append(row)
                log.info("Sigma v: {}".format(sigma_v))

            table = table_from_row_data(rows=tablerows)
            table["sigma_v"].unit = sigma_unit
            result[ch] = table

        return result
