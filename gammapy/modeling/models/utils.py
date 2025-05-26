# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import NoOverlapError
from astropy.time import Time
from astropy.visualization import quantity_support
from regions import PointSkyRegion
from gammapy.maps import HpxNDMap, Map, MapAxis, RegionNDMap
from gammapy.maps.hpx.io import HPX_FITS_CONVENTIONS, HpxConv
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_path, make_name
from gammapy.utils.time import time_ref_from_dict, time_ref_to_dict
from . import (
    LightCurveTemplateTemporalModel,
    Models,
    SkyModel,
    TemplateSpatialModel,
    SpectralModel,
    integrate_spectrum,
    scale_plot_flux,
)

__all__ = ["read_hermes_cube"]


def _template_model_from_cta_sdc(filename, t_ref=None):
    """To create a `LightCurveTemplateTemporalModel` from the energy-dependent temporal model files of the cta-sdc1.

    This format is subject to change.
    """
    filename = str(make_path(filename))
    with fits.open(filename) as hdul:
        frame = hdul[0].header.get("frame", "icrs")
        position = SkyCoord(
            hdul[0].header["LONG"] * u.deg, hdul[0].header["LAT"] * u.deg, frame=frame
        )

        energy_hdu = hdul["ENERGIES"]
        energy_axis = MapAxis.from_nodes(
            nodes=energy_hdu.data,
            unit=energy_hdu.header["TUNIT1"],
            name="energy",
            interp="log",
        )
        time_hdu = hdul["TIMES"]
        time_header = time_hdu.header

        if t_ref is None:
            t_ref = Time(55555.5, format="mjd", scale="tt")
        time_header.update(time_ref_to_dict(t_ref, t_ref.scale))
        time_min = time_hdu.data["Initial Time"]
        time_max = time_hdu.data["Final Time"]
        edges = np.append(time_min, time_max[-1]) * u.Unit(time_header["TUNIT1"])
        data = hdul["SPECTRA"]

        time_ref = time_ref_from_dict(time_header, scale=t_ref.scale)
        time_axis = MapAxis.from_edges(edges=edges, name="time", interp="log")

        reg_map = RegionNDMap.create(
            region=PointSkyRegion(center=position),
            axes=[energy_axis, time_axis],
            data=np.array(list(data.data) * u.Unit(data.header["UNITS"])),
        )
    return LightCurveTemplateTemporalModel(reg_map, t_ref=time_ref, filename=filename)


def read_hermes_cube(filename):
    """Read 3d templates produced with hermes."""
    # add hermes conventions to the list used by gammapy
    hermes_conv = HpxConv(
        convname="hermes-template",
        colstring="TFLOAT",
        hduname="xtension",
        frame="COORDTYPE",
        quantity_type="differential",
    )
    HPX_FITS_CONVENTIONS["hermes-template"] = hermes_conv

    maps = []
    energy_nodes = []
    with fits.open(filename) as hdulist:
        # cannot read directly in 3d with Map.read because BANDS HDU is missing
        # https://gamma-astro-data-formats.readthedocs.io/en/v0.2/skymaps/index.html#bands-hdu
        # so we have to loop over hdus and create the energy axis
        for hdu in hdulist[1:]:
            template = HpxNDMap.from_hdu(hdu, format="hermes-template")
            # fix missing/incompatible infos
            template._unit = u.Unit(hdu.header["TUNIT1"])  # .from_hdu expect "BUNIT"
            if template.geom.frame == "G":
                template._geom._frame = "galactic"
            maps.append(template)
            energy_nodes.append(hdu.header["ENERGY"])  # SI unit (see header comment)
    # create energy axis and set unit
    energy_nodes *= u.Joule
    energy_nodes = energy_nodes.to("GeV")
    axis = MapAxis(
        energy_nodes, interp="log", name="energy_true", node_type="center", unit="GeV"
    )
    return Map.from_stack(maps, axis=axis)


def cutout_template_models(models, cutout_kwargs, datasets_names=None):
    """Apply cutout to template models.

    Parameters
    ----------
    models : `~gammapy.modeling.Models`
        List of models
    cutout_kwargs : dict
        Arguments passed to `gammap.map.cutout`
    datasets_names : list of str
        Names of the datasets to which the new model is applied.

    Returns
    -------
    models_cut : `~gammapy.modeling.Models`
        Models with cutout
    """
    models_cut = Models()
    if models is None:
        return models_cut
    for m in models:
        if isinstance(m.spatial_model, TemplateSpatialModel):
            try:
                map_ = m.spatial_model.map.cutout(**cutout_kwargs)
            except (NoOverlapError, ValueError):
                continue
            template_cut = TemplateSpatialModel(
                map_,
                normalize=m.spatial_model.normalize,
            )
            model_cut = SkyModel(
                spatial_model=template_cut,
                spectral_model=m.spectral_model,
                datasets_names=datasets_names,
                name=m.name if not datasets_names else f"{m.name}_{make_name()}",
            )
            models_cut.append(model_cut)
        else:
            models_cut.append(m)
    return models_cut


def _get_model_parameters_samples(model, n_samples=10000, random_state=42):
    """Create SED samples from parameters and covariance using multivariate normal distribution.

    Parameters
    ----------
    model : `~gammapy.modeling.models.Model`
        Model used to compute the samples
    n_samples : int, optional
        Number of samples to generate. Default is 10000.
    random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}, optional
        Defines random number generator initialisation.
        Passed to `~gammapy.utils.random.get_random_state`. Default is 42.

    Returns
    -------
    samples : dict of `~astropy.units.Quantity`
        Dictionary of parameter samples.

    """
    result_samples = {}
    rng = get_random_state(random_state)

    params = model.parameters.free_parameters
    covar = model.covariance.get_subcovariance(params)

    samples = rng.multivariate_normal(
        params.value,
        covar.data,
        n_samples,
    )

    for i, par in enumerate(params):
        result_samples[par.name] = samples[:, i].T * par.unit

    for par in model.parameters:
        if par.name not in result_samples.keys():
            result_samples[par.name] = np.ones(n_samples) * par.quantity

    return result_samples


class FluxPredictionBand:
    """Evaluate flux of a spectral model based on samples of its parameters.

    Parameters
    ----------
    model : `~gammapy.modeling.models.SpectralModel`
        The spectral model.
    samples : dict
        The dictionary of samples. It must contain one key per model parameter name.
        The content of each key is the samples quantities for this parameter.
    """

    def __init__(self, model, samples):
        if not isinstance(model, SpectralModel):
            raise TypeError(
                f"PredictionBand requires SpectralModel. Got {type(model)} instead."
            )

        self._model, self._samples = self._validate(model, samples)

    @staticmethod
    def _validate(model, samples):
        names = set(model.parameters.names)
        samples_names = set(samples.keys())
        if not names == samples_names:
            raise ValueError(
                f"Sample dictionary does not match parameters: got {samples_names} instead of {names}"
            )

        sizes = [len(samples[_]) for _ in samples.keys()]
        if len(set(sizes)) > 1:
            raise ValueError("All parameter samples must have the same length.")

        return model, samples

    @property
    def model(self):
        return self._model

    @property
    def samples(self):
        return self._samples

    @staticmethod
    def _sigma_to_percentiles(n_sigma=1):
        """Return percentiles corresponding to -sigma, 0, sigma."""
        from scipy.stats import norm

        if n_sigma <= 0:
            raise ValueError(
                f"Number of sigma is expected to be positive float. Got {n_sigma} instead."
            )

        return 100 * (1 - norm.sf([-n_sigma, 0.0, n_sigma]))

    @staticmethod
    def _compute_dnde(energy, model, samples):
        return model.evaluate(energy[:, np.newaxis], **samples)

    @staticmethod
    def _compute_eflux(energy_min, energy_max, model, samples, ndecade=100):
        if hasattr(model, "evaluate_energy_flux"):
            return model.evaluate_energy_flux(
                energy_min[..., np.newaxis], energy_max[..., np.newaxis], **samples
            )
        else:
            return integrate_spectrum(
                model,
                energy_min,
                energy_max,
                ndecade=ndecade,
                parameter_samples=samples,
                eflux=True,
            )

    @staticmethod
    def _compute_flux(energy_min, energy_max, model, samples, ndecade=100):
        if hasattr(model, "evaluate_integral"):
            return model.evaluate_integral(
                energy_min[..., np.newaxis], energy_max[..., np.newaxis], **samples
            )
        else:
            res = integrate_spectrum(
                model,
                energy_min,
                energy_max,
                ndecade=ndecade,
                parameter_samples=samples,
            )
            return res

    @staticmethod
    def _compute_lo_hi(fluxes, n_sigma=1, axis=-1):
        percentiles = FluxPredictionBand._sigma_to_percentiles(n_sigma)

        flux_lo, flux_median, flux_hi = np.percentile(fluxes, percentiles, axis=axis)
        return flux_lo, flux_hi

    @staticmethod
    def _compute_asymetric_errors(fluxes, n_sigma=1, axis=-1):
        percentiles = FluxPredictionBand._sigma_to_percentiles(n_sigma)

        flux_min, flux_median, flux_max = np.percentile(fluxes, percentiles, axis=axis)
        errn = flux_median - flux_min
        errp = flux_max - flux_median

        return errn, errp

    def evaluate_error(self, energy, n_sigma=1):
        samples = self.model._convert_evaluate_unit(self.samples, energy)
        fluxes = self._compute_dnde(energy, self.model, samples)
        return self._compute_asymetric_errors(fluxes, n_sigma=n_sigma)

    def integral_error(self, energy_min, energy_max, n_sigma=1):
        samples = self.model._convert_evaluate_unit(self.samples, energy_min)
        fluxes = self._compute_flux(energy_min, energy_max, self.model, samples)
        return self._compute_asymetric_errors(fluxes, n_sigma=n_sigma)

    def energy_flux_error(self, energy_min, energy_max, n_sigma=1):
        samples = self.model._convert_evaluate_unit(self.samples, energy_min)
        fluxes = self._compute_eflux(energy_min, energy_max, self.model, samples)
        return self._compute_asymetric_errors(fluxes, n_sigma=n_sigma)

    @classmethod
    def from_model_covariance(cls, model, n_samples=10000, random_state=42):
        """Create random samples according to model covariance matrix.

        Assumes that samples are distributed following multivariate normal law.

        Parameters
        ----------
        model : `~gammapy.modeling.models.SpectralModel`
            Input spectral model.
        n_samples : int, optional
            Number of samples to generate. Default is 10000.
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}, optional
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`. Default is 42.
        """
        samples = _get_model_parameters_samples(
            model, n_samples=n_samples, random_state=random_state
        )
        return cls(model, samples)

    def _get_plot_flux_error(self, energy, sed_type):
        flux_lo = RegionNDMap.create(region=None, axes=[energy])
        flux_hi = RegionNDMap.create(region=None, axes=[energy])

        if sed_type in ["dnde", "norm"]:
            output = self.evaluate_error(energy.center)
        elif sed_type == "e2dnde":
            output = energy.center**2 * self.evaluate_error(energy.center)
        elif sed_type == "flux":
            output = self.integral_error(energy.edges_min, energy.edges_max)
        elif sed_type == "eflux":
            output = self.energy_flux_error(energy.edges_min, energy.edges_max)
        else:
            raise ValueError(f"Not a valid SED type: '{sed_type}'")

        flux_lo.quantity, flux_hi.quantity = output
        return flux_lo, flux_hi

    def plot_error(
        self,
        energy_bounds,
        ax=None,
        sed_type="dnde",
        energy_power=0,
        n_points=100,
        **kwargs,
    ):
        """Plot spectral model error band.

        .. note::

            This method calls ``ax.set_yscale("log", nonpositive='clip')`` and
            ``ax.set_xscale("log", nonposx='clip')`` to create a log-log representation.
            The additional argument ``nonposx='clip'`` avoids artefacts in the plot,
            when the error band extends to negative values (see also
            https://github.com/matplotlib/matplotlib/issues/8623).

            When you call ``plt.loglog()`` or ``plt.semilogy()`` explicitly in your
            plotting code and the error band extends to negative values, it is not
            shown correctly. To circumvent this issue also use
            ``plt.loglog(nonposx='clip', nonpositive='clip')``
            or ``plt.semilogy(nonpositive='clip')``.

        Parameters
        ----------
        energy_bounds : `~astropy.units.Quantity`, list of `~astropy.units.Quantity` or `~gammapy.maps.MapAxis`
            Energy bounds between which the model is to be plotted. Or an
            axis defining the energy bounds between which the model is to be plotted.
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        sed_type : {"dnde", "flux", "eflux", "e2dnde"}
            Evaluation methods of the model. Default is "dnde".
        energy_power : int, optional
            Power of energy to multiply flux axis with. Default is 0.
        n_points : int, optional
            Number of evaluation nodes. Default is 100.
        n_samples : int, optional
            Number of samples generated per parameter to estimate the error band. Default is 3500.
        **kwargs : dict
            Keyword arguments forwarded to `matplotlib.pyplot.fill_between`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes.

        Notes
        -----
        If ``energy_bounds`` is supplied as a list, tuple, or Quantity, an ``energy_axis`` is created internally with
        ``n_points`` bins between the given bounds.
        """
        from gammapy.estimators.map.core import DEFAULT_UNIT

        if self.model.is_norm_spectral_model:
            sed_type = "norm"

        if isinstance(energy_bounds, (tuple, list, u.Quantity)):
            energy_min, energy_max = energy_bounds
            energy = MapAxis.from_energy_bounds(
                energy_min,
                energy_max,
                n_points,
            )
        elif isinstance(energy_bounds, MapAxis):
            energy = energy_bounds

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("facecolor", "black")
        kwargs.setdefault("alpha", 0.2)
        kwargs.setdefault("linewidth", 0)

        if ax.yaxis.units is None:
            ax.yaxis.set_units(DEFAULT_UNIT[sed_type] * energy.unit**energy_power)

        flux_lo, flux_hi = self._get_plot_flux_error(sed_type=sed_type, energy=energy)
        y_lo = scale_plot_flux(flux_lo, energy_power).quantity[:, 0, 0]
        y_hi = scale_plot_flux(flux_hi, energy_power).quantity[:, 0, 0]

        with quantity_support():
            ax.fill_between(energy.center, y_lo, y_hi, **kwargs)

        self._plot_format_ax(ax, energy_power, sed_type)
        return ax
