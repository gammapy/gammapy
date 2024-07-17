# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spatial models."""
import logging
import os
import numpy as np
import scipy.integrate
import scipy.special
from scipy.interpolate import griddata
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord, angular_separation, position_angle
from astropy.nddata import NoOverlapError
from astropy.utils import lazyproperty
from regions import (
    CircleAnnulusSkyRegion,
    CircleSkyRegion,
    EllipseAnnulusSkyRegion,
    EllipseSkyRegion,
    PointSkyRegion,
    RectangleSkyRegion,
)
import matplotlib.pyplot as plt
from gammapy.maps import HpxNDMap, Map, MapCoord, WcsGeom, WcsNDMap
from gammapy.modeling import Parameter, Parameters
from gammapy.utils.compat import COPY_IF_NEEDED
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.interpolation import interpolation_scale
from gammapy.utils.regions import region_circle_to_ellipse, region_to_frame
from gammapy.utils.scripts import make_path
from .core import ModelBase, _build_parameters_from_dict

__all__ = [
    "ConstantFluxSpatialModel",
    "ConstantSpatialModel",
    "DiskSpatialModel",
    "GaussianSpatialModel",
    "GeneralizedGaussianSpatialModel",
    "PointSpatialModel",
    "Shell2SpatialModel",
    "ShellSpatialModel",
    "SpatialModel",
    "TemplateSpatialModel",
    "TemplateNDSpatialModel",
    "PiecewiseNormSpatialModel",
]


log = logging.getLogger(__name__)

MAX_OVERSAMPLING = 200


def compute_sigma_eff(lon_0, lat_0, lon, lat, phi, major_axis, e):
    """Effective radius, used for the evaluation of elongated models."""
    phi_0 = position_angle(lon_0, lat_0, lon, lat)
    d_phi = phi - phi_0
    minor_axis = Angle(major_axis * np.sqrt(1 - e**2))

    a2 = (major_axis * np.sin(d_phi)) ** 2
    b2 = (minor_axis * np.cos(d_phi)) ** 2
    denominator = np.sqrt(a2 + b2)
    sigma_eff = major_axis * minor_axis / denominator
    return minor_axis, sigma_eff


class SpatialModel(ModelBase):
    """Spatial model base class."""

    _type = "spatial"

    def __init__(self, **kwargs):
        frame = kwargs.pop("frame", "icrs")
        super().__init__(**kwargs)
        if not hasattr(self, "frame"):
            self.frame = frame

    def __call__(self, lon, lat, energy=None):
        """Call evaluate method."""
        kwargs = {par.name: par.quantity for par in self.parameters}

        if energy is None and self.is_energy_dependent:
            raise ValueError("Missing energy value for evaluation")

        if energy is not None:
            kwargs["energy"] = energy

        return self.evaluate(lon, lat, **kwargs)

    @property
    def evaluation_bin_size_min(self):
        return None

    # TODO: make this a hard-coded class attribute?
    @lazyproperty
    def is_energy_dependent(self):
        varnames = self.evaluate.__code__.co_varnames
        return "energy" in varnames

    @property
    def position(self):
        """Spatial model center position as a `~astropy.coordinates.SkyCoord`."""
        lon = self.lon_0.quantity
        lat = self.lat_0.quantity
        return SkyCoord(lon, lat, frame=self.frame)

    @position.setter
    def position(self, skycoord):
        """Spatial model center position."""
        coord = skycoord.transform_to(self.frame)
        self.lon_0.quantity = coord.data.lon
        self.lat_0.quantity = coord.data.lat

    @property
    def position_lonlat(self):
        """Spatial model center position `(lon, lat)` in radians and frame of the model."""
        lon = self.lon_0.quantity.to_value(u.rad)
        lat = self.lat_0.quantity.to_value(u.rad)
        return lon, lat

    # TODO: get rid of this!
    _phi_0 = 0.0

    @property
    def phi_0(self):
        return self._phi_0

    @phi_0.setter
    def phi_0(self, phi_0=0.0):
        self._phi_0 = phi_0

    @property
    def position_error(self):
        """Get 95% containment position error as `~regions.EllipseSkyRegion`."""
        if self.covariance is None:
            raise ValueError("No position error information available.")

        pars = self.parameters
        sub_covar = self.covariance.get_subcovariance(["lon_0", "lat_0"]).data.copy()
        cos_lat = np.cos(self.lat_0.quantity.to_value("rad"))
        sub_covar[0, 0] *= cos_lat**2.0
        sub_covar[0, 1] *= cos_lat
        sub_covar[1, 0] *= cos_lat
        eig_vals, eig_vecs = np.linalg.eig(sub_covar)
        lon_err, lat_err = np.sqrt(eig_vals)
        y_vec = eig_vecs[:, 0]
        phi = (np.arctan2(y_vec[1], y_vec[0]) * u.rad).to("deg") + self.phi_0
        err = np.sort([lon_err, lat_err])
        scale_r95 = Gauss2DPDF(sigma=1).containment_radius(0.95)
        err *= scale_r95
        if err[1] == lon_err * scale_r95:
            phi += 90 * u.deg
            height = 2 * err[1] * pars["lon_0"].unit
            width = 2 * err[0] * pars["lat_0"].unit
        else:
            height = 2 * err[1] * pars["lat_0"].unit
            width = 2 * err[0] * pars["lon_0"].unit

        return EllipseSkyRegion(
            center=self.position, height=height, width=width, angle=phi
        )

    def evaluate_geom(self, geom):
        """Evaluate model on `~gammapy.maps.Geom`.

        Parameters
        ----------
        geom : `~gammapy.maps.WcsGeom`
            Map geometry.

        Returns
        -------
        map : `~gammapy.maps.Map`
            Map containing the value in each spatial bin.
        """
        coords = geom.get_coord(frame=self.frame, sparse=True)

        if self.is_energy_dependent:
            return self(coords.lon, coords.lat, energy=coords["energy_true"])
        else:
            return self(coords.lon, coords.lat)

    def integrate_geom(self, geom, oversampling_factor=None):
        """Integrate model on `~gammapy.maps.Geom` or `~gammapy.maps.RegionGeom`.

        Integration is performed by simple rectangle approximation, the pixel center model value
        is multiplied by the pixel solid angle.
        An oversampling factor can be used for precision. By default, this parameter is set to None
        and an oversampling factor is automatically estimated based on the model estimation maximal
        bin width.

        For a RegionGeom, the model is integrated on a tangent WCS projection in the region.

        Parameters
        ----------
        geom : `~gammapy.maps.WcsGeom` or `~gammapy.maps.RegionGeom`
            The geom on which the integration is performed.
        oversampling_factor : int or None
            The oversampling factor to use for integration.
            Default is None: the factor is estimated from the model minimal bin size.

        Returns
        -------
        map : `~gammapy.maps.Map` or `gammapy.maps.RegionNDMap`
            Map containing the integral value in each spatial bin.
        """
        wcs_geom = geom
        mask = None

        if geom.is_region:
            wcs_geom = geom.to_wcs_geom().to_image()

        result = Map.from_geom(geom=wcs_geom)
        integrated = Map.from_geom(wcs_geom)

        pix_scale = np.max(wcs_geom.pixel_scales.to_value("deg"))
        if self.evaluation_radius is not None:
            try:
                width = 2 * np.maximum(
                    self.evaluation_radius.to_value("deg"), pix_scale
                )
                wcs_geom_cut = wcs_geom.cutout(self.position, width)
                integrated = Map.from_geom(wcs_geom_cut)
            except (NoOverlapError, ValueError):
                oversampling_factor = 1

        if oversampling_factor is None:
            if self.evaluation_bin_size_min is not None:
                res_scale = self.evaluation_bin_size_min.to_value("deg")
                if res_scale > 0:
                    oversampling_factor = np.minimum(
                        int(np.ceil(pix_scale / res_scale)), MAX_OVERSAMPLING
                    )
                else:
                    oversampling_factor = MAX_OVERSAMPLING
            else:
                oversampling_factor = 1

        if oversampling_factor > 1:
            upsampled_geom = integrated.geom.upsample(
                oversampling_factor, axis_name=None
            )
            # assume the upsampled solid angles are approximately factor**2 smaller
            values = self.evaluate_geom(upsampled_geom) / oversampling_factor**2
            upsampled = Map.from_geom(upsampled_geom, unit=values.unit)
            upsampled += values

            if geom.is_region:
                mask = geom.contains(upsampled_geom.get_coord()).astype("int")

            integrated.quantity = upsampled.downsample(
                oversampling_factor, preserve_counts=True, weights=mask
            ).quantity

            # Finally stack result
            result._unit = integrated.unit
            result.stack(integrated)
        else:
            values = self.evaluate_geom(wcs_geom)
            result._unit = values.unit
            result += values

        result *= result.geom.solid_angle()

        if geom.is_region:
            mask = result.geom.region_mask([geom.region])
            result = Map.from_geom(
                geom, data=np.sum(result.data[mask]), unit=result.unit
            )
        return result

    def to_dict(self, full_output=False):
        """Create dictionary for YAML serilisation."""
        data = super().to_dict(full_output)
        data["spatial"]["frame"] = self.frame
        data["spatial"]["parameters"] = data["spatial"].pop("parameters")
        return data

    @classmethod
    def from_dict(cls, data, **kwargs):
        """Create a spatial model from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing model parameters.
        kwargs : dict
            Keyword arguments passed to `~SpatialModel.from_parameters`.
        """
        kwargs = kwargs or {}
        spatial_data = data.get("spatial", data)
        if "frame" in spatial_data:
            kwargs["frame"] = spatial_data["frame"]
        return super().from_dict(data, **kwargs)

    @property
    def _evaluation_geom(self):
        if isinstance(self, TemplateSpatialModel):
            geom = self.map.geom
        else:
            width = 2 * max(self.evaluation_radius, 0.1 * u.deg)
            geom = WcsGeom.create(
                skydir=self.position, frame=self.frame, width=width, binsz=0.02
            )
        return geom

    def _get_plot_map(self, geom):
        if self.evaluation_radius is None and geom is None:
            raise ValueError(
                f"{self.__class__.__name__} requires geom to be defined for plotting."
            )

        if geom is None:
            geom = self._evaluation_geom

        data = self.evaluate_geom(geom)
        return Map.from_geom(geom, data=data.value, unit=data.unit)

    def plot(self, ax=None, geom=None, **kwargs):
        """Plot spatial model.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        geom : `~gammapy.maps.WcsGeom`, optional
            Geometry to use for plotting. Default is None.
        **kwargs : dict
            Keyword arguments passed to `~gammapy.maps.WcsMap.plot()`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes.
        """
        m = self._get_plot_map(geom)
        if not m.geom.is_flat:
            raise TypeError(
                "Use .plot_interactive() or .plot_grid() for Map dimension > 2"
            )
        return m.plot(ax=ax, **kwargs)

    def plot_interactive(self, ax=None, geom=None, **kwargs):
        """Plot spatial model.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        geom : `~gammapy.maps.WcsGeom`, optional
            Geom to use for plotting. Default is None.
        **kwargs : dict
            Keyword arguments passed to `~gammapy.maps.WcsMap.plot()`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes.
        """
        m = self._get_plot_map(geom)
        if m.geom.is_image:
            raise TypeError("Use .plot() for 2D Maps")
        m.plot_interactive(ax=ax, **kwargs)

    def plot_position_error(self, ax=None, **kwargs):
        """Plot position error.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes to plot the position error on. Default is None.
        **kwargs : dict
            Keyword arguments passed to `~gammapy.maps.WcsMap.plot()`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes.
        """
        # plot center position
        lon, lat = self.lon_0.value, self.lat_0.value

        ax = plt.gca() if ax is None else ax

        kwargs.setdefault("marker", "x")
        kwargs.setdefault("color", "red")
        kwargs.setdefault("label", "position")

        ax.scatter(lon, lat, transform=ax.get_transform(self.frame), **kwargs)

        # plot position error
        if not np.all(self.covariance.data == 0):
            region = self.position_error.to_pixel(ax.wcs)
            artist = region.as_artist(facecolor="none", edgecolor=kwargs["color"])
            ax.add_artist(artist)

        return ax

    def _to_region_error(self):
        pass

    def plot_error(
        self, ax=None, which="position", kwargs_position=None, kwargs_extension=None
    ):
        """Plot the errors of the spatial model.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes to plot the errors on. Default is None.
        which: list of str
            Which errors to plot.
            Available options are:

                * "all": all the optional steps are plotted
                * "position": plot the position error of the spatial model
                * "extension": plot the extension error of the spatial model

        kwargs_position : dict, optional
            Keyword arguments passed to `~SpatialModel.plot_position_error`.
            Default is None.
        kwargs_extension : dict, optional
            Keyword arguments passed to `~SpatialModel.plot_extension_error`.
            Default is None.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes.
        """
        kwargs_position = kwargs_position or {}
        kwargs_extension = kwargs_extension or {}

        ax = plt.gca() if ax is None else ax

        kwargs_extension.setdefault("edgecolor", "red")
        kwargs_extension.setdefault("facecolor", "red")
        kwargs_extension.setdefault("alpha", 0.15)
        kwargs_extension.setdefault("fill", True)

        if "all" in which:
            self.plot_position_error(ax, **kwargs_position)

            region = self._to_region_error()
            if region is not None:
                artist = region.to_pixel(ax.wcs).as_artist(**kwargs_extension)
                ax.add_artist(artist)

        if "position" in which:
            self.plot_position_error(ax, **kwargs_position)

        if "extension" in which:
            region = self._to_region_error()
            if region is not None:
                artist = region.to_pixel(ax.wcs).as_artist(**kwargs_extension)
                ax.add_artist(artist)

    def plot_grid(self, geom=None, **kwargs):
        """Plot spatial model energy slices in a grid.

        Parameters
        ----------
        geom : `~gammapy.maps.WcsGeom`, optional
            Geometry to use for plotting. Default is None.
        **kwargs : dict
            Keyword arguments passed to `~gammapy.maps.WcsMap.plot()`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes.
        """
        m = self._get_plot_map(geom)
        if (m.geom is None) or m.geom.is_image:
            raise TypeError("Use .plot() for 2D Maps")
        m.plot_grid(**kwargs)

    @classmethod
    def from_position(cls, position, **kwargs):
        """Define the position of the model using a `~astropy.coordinates.SkyCoord`.

        The model will be created in the frame of the `~astropy.coordinates.SkyCoord`.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position.

        Returns
        -------
        model : `SpatialModel`
            Spatial model.
        """
        lon_0, lat_0 = position.data.lon, position.data.lat
        return cls(lon_0=lon_0, lat_0=lat_0, frame=position.frame.name, **kwargs)

    @property
    def evaluation_radius(self):
        """Evaluation radius."""
        return None

    @property
    def evaluation_region(self):
        """Evaluation region."""
        if hasattr(self, "to_region"):
            return self.to_region()
        elif self.evaluation_radius is not None:
            return CircleSkyRegion(
                center=self.position,
                radius=self.evaluation_radius,
            )
        else:
            return None


class PointSpatialModel(SpatialModel):
    """Point Source.

    For more information see :ref:`point-spatial-model`.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position.
        Default is "0 deg", "0 deg".
    frame : {"icrs", "galactic"}
        Center position coordinate frame.
    """

    tag = ["PointSpatialModel", "point"]
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)

    @property
    def evaluation_bin_size_min(self):
        """Minimal evaluation bin size as an `~astropy.coordinates.Angle`."""
        return 0 * u.deg

    @property
    def evaluation_radius(self):
        """Evaluation radius as an `~astropy.coordinates.Angle`.

        Set as zero degrees.
        """
        return 0 * u.deg

    @staticmethod
    def _grid_weights(x, y, x0, y0):
        """Compute 4-pixel weights such that centroid is preserved."""
        dx = np.abs(x - x0)
        dx = np.where(dx < 1, 1 - dx, 0)

        dy = np.abs(y - y0)
        dy = np.where(dy < 1, 1 - dy, 0)

        return dx * dy

    @property
    def is_energy_dependent(self):
        return False

    def evaluate_geom(self, geom):
        """Evaluate model on `~gammapy.maps.Geom`."""
        values = self.integrate_geom(geom).data
        return values / geom.solid_angle()

    def integrate_geom(self, geom, oversampling_factor=None):
        """Integrate model on `~gammapy.maps.Geom`.

        Parameters
        ----------
        geom : `Geom`
            Map geometry.

        Returns
        -------
        flux : `Map`
            Predicted flux map.
        """
        geom_image = geom.to_image()
        if geom.is_hpx:
            idx, weights = geom_image.interp_weights({"skycoord": self.position})
            data = np.zeros(geom_image.data_shape)
            data[tuple(idx)] = weights
        else:
            x, y = geom_image.get_pix()
            x0, y0 = self.position.to_pixel(geom.wcs)
            data = self._grid_weights(x, y, x0, y0)
        return Map.from_geom(geom=geom_image, data=data, unit="")

    def to_region(self, **kwargs):
        """Model outline as a `~regions.PointSkyRegion`."""
        return PointSkyRegion(center=self.position, **kwargs)


class GaussianSpatialModel(SpatialModel):
    r"""Two-dimensional Gaussian model.

    For more information see :ref:`gaussian-spatial-model`.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position.
        Default is "0 deg", "0 deg".
    sigma : `~astropy.coordinates.Angle`
        Length of the major semiaxis of the Gaussian, in angular units.
        Default is 1 deg.
    e : `float`
        Eccentricity of the Gaussian (:math:`0< e< 1`).
        Default is 0.
    phi : `~astropy.coordinates.Angle`
        Rotation angle :math:`\phi`: of the major semiaxis.
        Increases counter-clockwise from the North direction.
        Default is 0 deg.
    frame : {"icrs", "galactic"}
        Center position coordinate frame.
    """

    tag = ["GaussianSpatialModel", "gauss"]

    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)
    sigma = Parameter("sigma", "1 deg", min=0)
    e = Parameter("e", 0, min=0, max=1, frozen=True)
    phi = Parameter("phi", "0 deg", frozen=True)

    @property
    def evaluation_bin_size_min(self):
        """Minimal evaluation bin size as an `~astropy.coordinates.Angle`, chosen as sigma/3."""
        return self.parameters["sigma"].quantity / 3.0

    @property
    def evaluation_radius(self):
        r"""Evaluation radius as an `~astropy.coordinates.Angle`.

        Set as :math:`5\sigma`.
        """
        return 5 * self.parameters["sigma"].quantity

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, sigma, e, phi):
        """Evaluate model."""
        sep = angular_separation(lon, lat, lon_0, lat_0)

        if e == 0:
            a = 1.0 - np.cos(sigma)
            norm = (1 / (4 * np.pi * a * (1.0 - np.exp(-1.0 / a)))).value
        else:
            minor_axis, sigma_eff = compute_sigma_eff(
                lon_0, lat_0, lon, lat, phi, sigma, e
            )
            a = 1.0 - np.cos(sigma_eff)
            norm = (1 / (2 * np.pi * sigma * minor_axis)).to_value("sr-1")

        exponent = -0.5 * ((1 - np.cos(sep)) / a)
        return u.Quantity(norm * np.exp(exponent).value, "sr-1", copy=COPY_IF_NEEDED)

    def to_region(self, x_sigma=1.5, **kwargs):
        r"""Model outline at a given number of :math:`\sigma`.

        Parameters
        ----------
        x_sigma : float
            Number of :math:`\sigma
            Default is :math:`1.5\sigma` which corresponds to about 68%
            containment for a 2D symmetric Gaussian.

        Returns
        -------
        region : `~regions.EllipseSkyRegion`
            Model outline.
        """
        minor_axis = Angle(self.sigma.quantity * np.sqrt(1 - self.e.quantity**2))
        return EllipseSkyRegion(
            center=self.position,
            height=2 * x_sigma * self.sigma.quantity,
            width=2 * x_sigma * minor_axis,
            angle=self.phi.quantity,
            **kwargs,
        )

    @property
    def evaluation_region(self):
        """Evaluation region consistent with evaluation radius."""
        return self.to_region(x_sigma=5)

    def _to_region_error(self, x_sigma=1.5):
        r"""Plot model error at a given number of :math:`\sigma`.

        Parameters
        ----------
        x_sigma : float
            Number of :math:`\sigma
            Default is :math:`1.5\sigma` which corresponds to about 68%
            containment for a 2D symmetric Gaussian.

        Returns
        -------
        region : `~regions.EllipseSkyRegion`
            Model error region.
        """
        sigma_hi = self.sigma.quantity + (self.sigma.error * self.sigma.unit)
        sigma_lo = self.sigma.quantity - (self.sigma.error * self.sigma.unit)

        minor_axis_hi = Angle(
            sigma_hi * np.sqrt(1 - (self.e.quantity + self.e.error) ** 2)
        )
        minor_axis_lo = Angle(
            sigma_lo * np.sqrt(1 - (self.e.quantity - self.e.error) ** 2)
        )

        return EllipseAnnulusSkyRegion(
            center=self.position,
            inner_height=2 * x_sigma * sigma_lo,
            outer_height=2 * x_sigma * sigma_hi,
            inner_width=2 * x_sigma * minor_axis_lo,
            outer_width=2 * x_sigma * minor_axis_hi,
            angle=self.phi.quantity,
        )


class GeneralizedGaussianSpatialModel(SpatialModel):
    r"""Two-dimensional Generalized Gaussian model.

    For more information see :ref:`generalized-gaussian-spatial-model`.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position.
        Default is "0 deg", "0 deg".
    r_0 : `~astropy.coordinates.Angle`
        Length of the major semiaxis, in angular units.
        Default is 1 deg.
    eta : `float`
        Shape parameter within (0, 1). Special cases for disk: ->0, Gaussian: 0.5, Laplace:1
        Default is 0.5.
    e : `float`
        Eccentricity (:math:`0< e< 1`).
        Default is 0.
    phi : `~astropy.coordinates.Angle`
        Rotation angle :math:`\phi`: of the major semiaxis.
        Increases counter-clockwise from the North direction.
        Default is 0 deg.
    frame : {"icrs", "galactic"}
        Center position coordinate frame.
    """

    tag = ["GeneralizedGaussianSpatialModel", "gauss-general"]
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)
    r_0 = Parameter("r_0", "1 deg")
    eta = Parameter("eta", 0.5, min=0.01, max=1.0)
    e = Parameter("e", 0.0, min=0.0, max=1.0, frozen=True)
    phi = Parameter("phi", "0 deg", frozen=True)

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, r_0, eta, e, phi):
        sep = angular_separation(lon, lat, lon_0, lat_0)
        if isinstance(eta, u.Quantity):
            eta = eta.value  # gamma function does not allow quantities
        minor_axis, r_eff = compute_sigma_eff(lon_0, lat_0, lon, lat, phi, r_0, e)
        z = sep / r_eff
        norm = 1 / (2 * np.pi * minor_axis * r_0 * eta * scipy.special.gamma(2 * eta))
        return (norm * np.exp(-(z ** (1 / eta)))).to("sr-1")

    @property
    def evaluation_bin_size_min(self):
        """Minimal evaluation bin size as an `~astropy.coordinates.Angle`.

        The bin min size is defined as r_0/(3+8*eta)/(e+1).
        """
        return self.r_0.quantity / (3 + 8 * self.eta.value) / (self.e.value + 1)

    @property
    def evaluation_radius(self):
        r"""Evaluation radius as an `~astropy.coordinates.Angle`.

        The evaluation radius is defined as r_eval = r_0*(1+8*eta) so it verifies:
        r_eval -> r_0 if eta -> 0
        r_eval = 5*r_0 > 5*sigma_gauss = 5*r_0/sqrt(2) ~ 3.5*r_0 if eta=0.5
        r_eval = 9*r_0 > 5*sigma_laplace = 5*sqrt(2)*r_0 ~ 7*r_0 if eta = 1
        r_eval -> inf if eta -> inf
        """
        return self.r_0.quantity * (1 + 8 * self.eta.value)

    def to_region(self, x_r_0=1, **kwargs):
        r"""Model outline at a given number of :math:`r_0`.

        Parameters
        ----------
        x_r_0 : float, optional
            Number of :math:`r_0`. Default is 1.

        Returns
        -------
        region : `~regions.EllipseSkyRegion`
            Model outline.
        """
        minor_axis = Angle(self.r_0.quantity * np.sqrt(1 - self.e.quantity**2))
        return EllipseSkyRegion(
            center=self.position,
            height=2 * x_r_0 * self.r_0.quantity,
            width=2 * x_r_0 * minor_axis,
            angle=self.phi.quantity,
            **kwargs,
        )

    @property
    def evaluation_region(self):
        """Evaluation region consistent with evaluation radius."""
        scale = self.evaluation_radius / self.r_0.quantity
        return self.to_region(x_r_0=scale)

    def _to_region_error(self, x_r_0=1):
        r"""Model error at a given number of :math:`r_0`.

        Parameters
        ----------
        x_r_0 : float, optional
            Number of :math:`r_0`. Default is 1.

        Returns
        -------
        region : `~regions.EllipseSkyRegion`
            Model error region.
        """
        r_0_lo = self.r_0.quantity - self.r_0.error * self.r_0.unit
        r_0_hi = self.r_0.quantity + self.r_0.error * self.r_0.unit

        minor_axis_hi = Angle(
            r_0_hi * np.sqrt(1 - (self.e.quantity + self.e.error) ** 2)
        )
        minor_axis_lo = Angle(
            r_0_lo * np.sqrt(1 - (self.e.quantity - self.e.error) ** 2)
        )

        return EllipseAnnulusSkyRegion(
            center=self.position,
            inner_height=2 * x_r_0 * r_0_lo,
            outer_height=2 * x_r_0 * r_0_hi,
            inner_width=2 * x_r_0 * minor_axis_lo,
            outer_width=2 * x_r_0 * minor_axis_hi,
            angle=self.phi.quantity,
        )


class DiskSpatialModel(SpatialModel):
    r"""Constant disk model.

    For more information see :ref:`disk-spatial-model`.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position.
        Default is "0 deg", "0 deg".
    r_0 : `~astropy.coordinates.Angle`
        :math:`a`: length of the major semiaxis, in angular units.
        Default is 1 deg.
    e : `float`
        Eccentricity of the ellipse (:math:`0< e< 1`).
        Default is 0.
    phi : `~astropy.coordinates.Angle`
        Rotation angle :math:`\phi`: of the major semiaxis.
        Increases counter-clockwise from the North direction.
        Default is 0 deg.
    edge_width : float
        Width of the edge. The width is defined as the range within which
        the smooth edge of the model drops from 95% to 5% of its amplitude.
        It is given as fraction of r_0.
        Default is 0.01.
    frame : {"icrs", "galactic"}
        Center position coordinate frame.
    """

    tag = ["DiskSpatialModel", "disk"]
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)
    r_0 = Parameter("r_0", "1 deg", min=0)
    e = Parameter("e", 0, min=0, max=1, frozen=True)
    phi = Parameter("phi", "0 deg", frozen=True)
    edge_width = Parameter("edge_width", value=0.01, min=0, max=1, frozen=True)

    @property
    def evaluation_bin_size_min(self):
        """Minimal evaluation bin size as an `~astropy.coordinates.Angle`.

        The bin min size is defined as r_0*(1-edge_width)/10.
        """
        return self.r_0.quantity * (1 - self.edge_width.quantity) / 10.0

    @property
    def evaluation_radius(self):
        """Evaluation radius as an `~astropy.coordinates.Angle`.

        Set to the length of the semi-major axis plus the edge width.
        """
        return 1.1 * self.r_0.quantity * (1 + self.edge_width.quantity)

    @staticmethod
    def _evaluate_norm_factor(r_0, e):
        """Compute the normalization factor."""
        semi_minor = r_0 * np.sqrt(1 - e**2)

        def integral_fcn(x, a, b):
            A = 1 / np.sin(a) ** 2
            B = 1 / np.sin(b) ** 2
            C = A - B
            cs2 = np.cos(x) ** 2

            return 1 - np.sqrt(1 - 1 / (B + C * cs2))

        return (
            2
            * scipy.integrate.quad(
                lambda x: integral_fcn(x, r_0, semi_minor), 0, np.pi
            )[0]
        ) ** -1

    @staticmethod
    def _evaluate_smooth_edge(x, width):
        value = (x / width).to_value("")
        edge_width_95 = 2.326174307353347
        return 0.5 * (1 - scipy.special.erf(value * edge_width_95))

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, r_0, e, phi, edge_width):
        """Evaluate model."""
        sep = angular_separation(lon, lat, lon_0, lat_0)

        if e == 0:
            sigma_eff = r_0
        else:
            sigma_eff = compute_sigma_eff(lon_0, lat_0, lon, lat, phi, r_0, e)[1]

        norm = DiskSpatialModel._evaluate_norm_factor(r_0, e)

        in_ellipse = DiskSpatialModel._evaluate_smooth_edge(
            sep - sigma_eff, sigma_eff * edge_width
        )
        return u.Quantity(norm * in_ellipse, "sr-1", copy=COPY_IF_NEEDED)

    def to_region(self, **kwargs):
        """Model outline as a `~regions.EllipseSkyRegion`."""
        minor_axis = Angle(self.r_0.quantity * np.sqrt(1 - self.e.quantity**2))
        return EllipseSkyRegion(
            center=self.position,
            height=2 * self.r_0.quantity,
            width=2 * minor_axis,
            angle=self.phi.quantity,
            **kwargs,
        )

    @classmethod
    def from_region(cls, region, **kwargs):
        """Create a `DiskSpatialModel from a ~regions.EllipseSkyRegion`.

        Parameters
        ----------
        region : `~regions.EllipseSkyRegion` or ~regions.CircleSkyRegion`
            Region to create model from.
        kwargs : dict
            Keyword arguments passed to `~gammapy.modeling.models.DiskSpatialModel`.

        Returns
        -------
        spatial_model : `~gammapy.modeling.models.DiskSpatialModel`
            Spatial model.
        """
        if isinstance(region, CircleSkyRegion):
            region = region_circle_to_ellipse(region)
        if not isinstance(region, EllipseSkyRegion):
            raise ValueError(
                f"Please provide a `CircleSkyRegion` "
                f"or `EllipseSkyRegion`, got {type(region)} instead."
            )
        frame = kwargs.pop("frame", region.center.frame)
        region = region_to_frame(region, frame=frame)

        if region.height > region.width:
            major_axis, minor_axis = region.height, region.width
            phi = region.angle
        else:
            minor_axis, major_axis = region.height, region.width
            phi = 90 * u.deg + region.angle

        kwargs.setdefault("phi", phi)
        kwargs.setdefault("e", np.sqrt(1.0 - np.power(minor_axis / major_axis, 2)))
        kwargs.setdefault("r_0", major_axis / 2.0)

        return cls.from_position(region.center, **kwargs)

    def _to_region_error(self):
        """Model error.

        Returns
        -------
        region : `~regions.EllipseSkyRegion`
            Model error region.
        """
        r_0_lo = self.r_0.quantity - self.r_0.error * self.r_0.unit
        r_0_hi = self.r_0.quantity + self.r_0.error * self.r_0.unit

        minor_axis_hi = Angle(
            r_0_hi * np.sqrt(1 - (self.e.quantity + self.e.error) ** 2)
        )
        minor_axis_lo = Angle(
            r_0_lo * np.sqrt(1 - (self.e.quantity - self.e.error) ** 2)
        )

        return EllipseAnnulusSkyRegion(
            center=self.position,
            inner_height=2 * r_0_lo,
            outer_height=2 * r_0_hi,
            inner_width=2 * minor_axis_lo,
            outer_width=2 * minor_axis_hi,
            angle=self.phi.quantity,
        )


class ShellSpatialModel(SpatialModel):
    r"""Shell model.

    For more information see :ref:`shell-spatial-model`.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position.
        Default is "0 deg", "0 deg".
    radius : `~astropy.coordinates.Angle`
        Inner radius, :math:`r_{in}`.
        Default is 1 deg.
    width : `~astropy.coordinates.Angle`
        Shell width.
        Default is 0.2 deg.
    frame : {"icrs", "galactic"}
        Center position coordinate frame.

    See Also
    --------
    Shell2SpatialModel
    """

    tag = ["ShellSpatialModel", "shell"]
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)
    radius = Parameter("radius", "1 deg")
    width = Parameter("width", "0.2 deg")

    @property
    def evaluation_bin_size_min(self):
        """Minimal evaluation bin size as an `~astropy.coordinates.Angle`.

        The bin min size is defined as the shell width.
        """
        return self.width.quantity

    @property
    def evaluation_radius(self):
        r"""Evaluation radius as an `~astropy.coordinates.Angle`.

        Set to :math:`r_\text{out}`.
        """
        return self.radius.quantity + self.width.quantity

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, radius, width):
        """Evaluate model."""
        sep = angular_separation(lon, lat, lon_0, lat_0)
        radius_out = radius + width

        norm = 3 / (2 * np.pi * (radius_out**3 - radius**3))

        with np.errstate(invalid="ignore"):
            # np.where and np.select do not work with quantities, so we use the
            # workaround with indexing
            value = np.sqrt(radius_out**2 - sep**2)
            mask = sep < radius
            value[mask] = (value - np.sqrt(radius**2 - sep**2))[mask]
            value[sep > radius_out] = 0

        return norm * value

    def to_region(self, **kwargs):
        """Model outline as a `~regions.CircleAnnulusSkyRegion`."""
        return CircleAnnulusSkyRegion(
            center=self.position,
            inner_radius=self.radius.quantity,
            outer_radius=self.radius.quantity + self.width.quantity,
            **kwargs,
        )


class Shell2SpatialModel(SpatialModel):
    r"""Shell model with outer radius and relative width parametrization.

    For more information see :ref:`shell2-spatial-model`.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position.
        Default is "0 deg", "0 deg".
    r_0 : `~astropy.coordinates.Angle`
        Outer radius, :math:`r_{out}`.
        Default is 1 deg.
    eta : float
        Shell width relative to outer radius, r_0, should be within (0,1).
        Default is 0.2.
    frame : {"icrs", "galactic"}
        Center position coordinate frame.

    See Also
    --------
    ShellSpatialModel
    """

    tag = ["Shell2SpatialModel", "shell2"]
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)
    r_0 = Parameter("r_0", "1 deg")
    eta = Parameter("eta", 0.2, min=0.02, max=1)

    @property
    def evaluation_bin_size_min(self):
        """Minimal evaluation bin size as an `~astropy.coordinates.Angle`.

        The bin min size is defined as r_0*eta.
        """
        return self.eta.value * self.r_0.quantity

    @property
    def evaluation_radius(self):
        r"""Evaluation radius as an `~astropy.coordinates.Angle`.

        Set to :math:`r_\text{out}`.
        """
        return self.r_0.quantity

    @property
    def r_in(self):
        return (1 - self.eta.quantity) * self.r_0.quantity

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, r_0, eta):
        """Evaluate model."""
        sep = angular_separation(lon, lat, lon_0, lat_0)
        r_in = (1 - eta) * r_0

        norm = 3 / (2 * np.pi * (r_0**3 - r_in**3))

        with np.errstate(invalid="ignore"):
            # np.where and np.select do not work with quantities, so we use the
            # workaround with indexing
            value = np.sqrt(r_0**2 - sep**2)
            mask = sep < r_in
            value[mask] = (value - np.sqrt(r_in**2 - sep**2))[mask]
            value[sep > r_0] = 0

        return norm * value

    def to_region(self, **kwargs):
        """Model outline as a `~regions.CircleAnnulusSkyRegion`."""
        return CircleAnnulusSkyRegion(
            center=self.position,
            inner_radius=self.r_in,
            outer_radius=self.r_0.quantity,
            **kwargs,
        )


class ConstantSpatialModel(SpatialModel):
    """Spatially constant (isotropic) spatial model.

    For more information see :ref:`constant-spatial-model`.

    Parameters
    ----------
    value : `~astropy.units.Quantity`
        Value. Default is 1 sr-1.
    """

    tag = ["ConstantSpatialModel", "const"]
    value = Parameter("value", "1 sr-1", frozen=True)

    frame = "icrs"
    evaluation_radius = None
    position = None

    def to_dict(self, full_output=False):
        """Create dictionary for YAML serilisation."""
        # redefined to ignore frame attribute from parent class
        data = super().to_dict(full_output)
        data["spatial"].pop("frame")
        data["spatial"]["parameters"] = []
        return data

    @staticmethod
    def evaluate(lon, lat, value):
        """Evaluate model."""
        return value

    def to_region(self, **kwargs):
        """Model outline as a `~regions.RectangleSkyRegion`."""
        return RectangleSkyRegion(
            center=SkyCoord(0 * u.deg, 0 * u.deg, frame=self.frame),
            height=180 * u.deg,
            width=360 * u.deg,
            **kwargs,
        )


class ConstantFluxSpatialModel(SpatialModel):
    """Spatially constant flux spatial model.

    For more information see :ref:`constant-spatial-model`.

    """

    tag = ["ConstantFluxSpatialModel", "const-flux"]

    frame = "icrs"
    evaluation_radius = None
    position = None

    def to_dict(self, full_output=False):
        """Create dictionary for YAML serilisation."""
        # redefined to ignore frame attribute from parent class
        data = super().to_dict(full_output)
        data["spatial"].pop("frame")
        return data

    @staticmethod
    def evaluate(lon, lat):
        """Evaluate model."""
        return 1 / u.sr

    @staticmethod
    def evaluate_geom(geom):
        """Evaluate model."""
        return 1 / geom.solid_angle()

    @staticmethod
    def integrate_geom(geom, oversampling_factor=None):
        """Evaluate model."""
        return Map.from_geom(geom=geom, data=1)

    def to_region(self, **kwargs):
        """Model outline as a `~regions.RectangleSkyRegion`."""
        return RectangleSkyRegion(
            center=SkyCoord(0 * u.deg, 0 * u.deg, frame=self.frame),
            height=180 * u.deg,
            width=360 * u.deg,
            **kwargs,
        )


class TemplateSpatialModel(SpatialModel):
    """Spatial sky map template model.

    For more information see :ref:`template-spatial-model`.
    By default, the position of the model is fixed at the center of the map.
    The position can be fitted by unfreezing the `lon_0` and `lat_0` parameters.
    In that case, the coordinate of every pixel is shifted in lon and lat
    in the frame of the map. NOTE: planar distances are calculated, so
    the results are correct only when the fitted position is close to the
    map center.

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Map template.
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialisation.
    normalize : bool
        Normalize the input map so that it integrates to unity.
    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_coord`.
        Default arguments are {'method': 'linear', 'fill_value': 0, "values_scale": "log"}.
    filename : str
        Name of the map file.
    copy_data : bool
        Create a deepcopy of the map data or directly use the original. Default is True.
        Use False to save memory in case of large maps.
    **kwargs : dict
        Keyword arguments forwarded to `SpatialModel.__init__`.
    """

    tag = ["TemplateSpatialModel", "template"]
    lon_0 = Parameter("lon_0", np.nan, unit="deg", frozen=True)
    lat_0 = Parameter("lat_0", np.nan, unit="deg", min=-90, max=90, frozen=True)

    def __init__(
        self,
        map,
        meta=None,
        normalize=True,
        interp_kwargs=None,
        filename=None,
        copy_data=True,
        **kwargs,
    ):
        if (map.data < 0).any():
            log.warning("Map has negative values. Check and fix this!")
        if (map.data == 0.0).all():
            log.warning("Map values are all zeros. Check and fix this!")
        if not map.geom.is_image and (map.data.sum(axis=(1, 2)) == 0).any():
            log.warning(
                "Map values are all zeros in at least one energy bin. Check and fix this!"
            )

        if filename is not None:
            filename = str(make_path(filename))

        self.normalize = normalize

        if normalize:
            # Normalize the diffuse map model so that it integrates to unity
            if map.geom.is_image:
                data_sum = map.data.sum()
            else:
                # Normalize in each energy bin
                data_sum = map.data.sum(axis=(1, 2)).reshape((-1, 1, 1))

            data = np.divide(
                map.data.astype(float),
                data_sum,
                out=np.zeros_like(map.data, dtype=float),
                where=data_sum != 0,
            )
            data /= map.geom.solid_angle().to_value("sr")
            map = map.copy(data=data, unit="sr-1")

        if map.unit.is_equivalent(""):
            map = map.copy(data=map.data, unit="sr-1")
            log.warning("Missing spatial template unit, assuming sr^-1")

        if copy_data:
            self._map = map.copy()
        else:
            self._map = map.copy(data=map.data)

        self.meta = {} if meta is None else meta

        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        interp_kwargs.setdefault("method", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        interp_kwargs.setdefault("values_scale", "log")

        self._interp_kwargs = interp_kwargs
        self.filename = filename
        kwargs["frame"] = self.map.geom.frame
        if "lon_0" not in kwargs or (
            isinstance(kwargs["lon_0"], Parameter) and np.isnan(kwargs["lon_0"].value)
        ):
            kwargs["lon_0"] = self.map_center.data.lon
        if "lat_0" not in kwargs or (
            isinstance(kwargs["lat_0"], Parameter) and np.isnan(kwargs["lat_0"].value)
        ):
            kwargs["lat_0"] = self.map_center.data.lat
        super().__init__(**kwargs)

    def __str__(self):
        width = self.map.geom.width
        data_min = np.nanmin(self.map.data)
        data_max = np.nanmax(self.map.data)

        prnt = (
            f"{self.__class__.__name__} model summary:\n "
            f"Model center: {self.position} \n "
            f"Map center: {self.map_center} \n "
            f"Map width: {width} \n "
            f"Data min: {data_min} \n"
            f"Data max: {data_max} \n"
            f"Data unit: {self.map.unit}"
        )

        if self.is_energy_dependent:
            energy_min = self.map.geom.axes["energy_true"].center[0]
            energy_max = self.map.geom.axes["energy_true"].center[-1]
            prnt1 = f"Energy min: {energy_min} \n" f"Energy max: {energy_max} \n"
            prnt = prnt + prnt1

        return prnt

    def copy(self, copy_data=False, **kwargs):
        """Copy model.

        Parameters
        ----------
        copy_data : bool
            Whether to copy the data. Default is False.
        **kwargs : dict
            Keyword arguments forwarded to `TemplateSpatialModel`.

        Returns
        -------
        model : `TemplateSpatialModel`
            Copied template spatial model.
        """
        kwargs.setdefault("map", self.map)
        kwargs.setdefault("meta", self.meta.copy())
        kwargs.setdefault("normalize", self.normalize)
        kwargs.setdefault("interp_kwargs", self._interp_kwargs)
        kwargs.setdefault("filename", self.filename)
        kwargs.setdefault("lon_0", self.parameters["lon_0"].copy())
        kwargs.setdefault("lat_0", self.parameters["lat_0"].copy())
        return self.__class__(copy_data=copy_data, **kwargs)

    @property
    def map(self):
        """Template map as a `~gammapy.maps.Map` object."""
        return self._map

    @property
    def is_energy_dependent(self):
        return "energy_true" in self.map.geom.axes.names

    @property
    def evaluation_radius(self):
        """Evaluation radius as an `~astropy.coordinates.Angle`.

        Set to half of the maximal dimension of the map.
        """
        return np.max(self.map.geom.width) / 2.0

    @property
    def map_center(self):
        return self.map.geom.center_skydir

    @classmethod
    def read(cls, filename, normalize=True, **kwargs):
        """Read spatial template model from FITS image.

        If unit is not given in the FITS header the default is ``sr-1``.

        Parameters
        ----------
        filename : str
            FITS image filename.
        normalize : bool
            Normalize the input map so that it integrates to unity.
        kwargs : dict
            Keyword arguments passed to `Map.read()`.
        """
        m = Map.read(filename, **kwargs)
        return cls(m, normalize=normalize, filename=filename)

    def evaluate(self, lon, lat, energy=None, lon_0=None, lat_0=None):
        """Evaluate the model at given coordinates.

        Note that, if the map data assume negative values, these are
        clipped to zero.
        """

        offset_lon = 0.0 * u.deg if lon_0 is None else lon_0 - self.map_center.data.lon
        offset_lat = 0.0 * u.deg if lat_0 is None else lat_0 - self.map_center.data.lat

        coord = {
            "lon": (lon - offset_lon).to_value("deg"),
            "lat": (lat - offset_lat).to_value("deg"),
        }
        if energy is not None:
            coord["energy_true"] = energy

        val = self.map.interp_by_coord(coord, **self._interp_kwargs)
        val = np.clip(val, 0, a_max=None)
        return u.Quantity(val, self.map.unit, copy=COPY_IF_NEEDED)

    @property
    def position_lonlat(self):
        """Spatial model center position `(lon, lat)` in radians and frame of the model."""
        lon = self.position.data.lon.rad
        lat = self.position.data.lat.rad
        return lon, lat

    @classmethod
    def from_dict(cls, data):
        data = data["spatial"]
        filename = data["filename"]
        normalize = data.get("normalize", True)
        m = Map.read(filename)
        pars = data.get("parameters")
        if pars is not None:
            parameters = _build_parameters_from_dict(pars, cls.default_parameters)
            kwargs = {par.name: par for par in parameters}
        else:
            kwargs = {}
        return cls(m, normalize=normalize, filename=filename, **kwargs)

    def to_dict(self, full_output=False):
        """Create dictionary for YAML serilisation."""
        data = super().to_dict(full_output)
        data["spatial"]["filename"] = self.filename
        data["spatial"]["normalize"] = self.normalize
        data["spatial"]["unit"] = str(self.map.unit)
        return data

    def write(self, overwrite=False, filename=None):
        """
        Write the map.

        Parameters
        ----------
        overwrite: bool, optional
            Overwrite existing file.
            Default is False, which will raise a warning if the template file exists already.
        filename: str, optional
            Filename of the template model. By default, the template model
            will be saved with the `TemplateSpatialModel.filename` attribute,
            if `filename` is provided this attribute will be updated.
        """
        if filename is not None:
            self.filename = filename
        if self.filename is None:
            raise IOError("Missing filename")
        if os.path.isfile(make_path(self.filename)) and not overwrite:
            log.warning("Template file already exits, and overwrite is False")
        else:
            self.map.write(self.filename, overwrite=overwrite)

    def to_region(self, **kwargs):
        """Model outline from template map boundary as a `~regions.RectangleSkyRegion`."""
        return RectangleSkyRegion(
            center=self.map.geom.center_skydir,
            width=self.map.geom.width[0][0],
            height=self.map.geom.width[1][0],
            **kwargs,
        )

    def plot(self, ax=None, geom=None, **kwargs):
        if geom is None:
            geom = self.map.geom
        super().plot(ax=ax, geom=geom, **kwargs)

    def plot_interactive(self, ax=None, geom=None, **kwargs):
        if geom is None:
            geom = self.map.geom
        super().plot_interactive(ax=ax, geom=geom, **kwargs)


class TemplateNDSpatialModel(SpatialModel):
    """A model generated from a ND map where extra dimensions define the parameter space.

    Parameters
    ----------
    map : `~gammapy.maps.WcsNDMap` or `~gammapy.maps.HpxNDMap`
        Map template.
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialisation.
    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_pix`.
        Default arguments are {'method': 'linear', 'fill_value': 0, "values_scale": "log"}.
    copy_data : bool
        Create a deepcopy of the map data or directly use the original. Default is True.
        Use False to save memory in case of large maps.

    """

    tag = ["TemplateNDSpatialModel", "templateND"]

    def __init__(
        self,
        map,
        interp_kwargs=None,
        meta=None,
        filename=None,
        copy_data=True,
    ):
        if not isinstance(map, (HpxNDMap, WcsNDMap)):
            raise TypeError("Map should be a HpxNDMap or WcsNDMap")
        if copy_data:
            self._map = map.copy()
        else:
            self._map = map.copy(data=map.data)
        self.meta = dict() if meta is None else meta
        if filename is not None:
            filename = str(make_path(filename))
        self.filename = filename

        parameters = []
        for axis in map.geom.axes:
            if axis.name not in ["energy_true", "energy"]:
                center = (axis.bounds[1] + axis.bounds[0]) / 2
                parameter = Parameter(
                    name=axis.name,
                    value=center,
                    unit=axis.unit,
                    scale_method="scale10",
                    min=axis.bounds[0],
                    max=axis.bounds[-1],
                    interp=axis.interp,
                )
                parameters.append(parameter)
        self.default_parameters = Parameters(parameters)

        interp_kwargs = interp_kwargs or {}
        interp_kwargs.setdefault("method", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        interp_kwargs.setdefault("values_scale", "log")
        self._interp_kwargs = interp_kwargs
        super().__init__()

    @property
    def map(self):
        """Template map as a `~gammapy.maps.WcsNDMap` or `~gammapy.maps.HpxNDMap`."""
        return self._map

    @property
    def is_energy_dependent(self):
        return "energy_true" in self.map.geom.axes.names

    def evaluate(self, lon, lat, energy=None, **kwargs):
        coord = {
            "lon": lon.to_value("deg"),
            "lat": lat.to_value("deg"),
        }
        if energy is not None:
            coord["energy_true"] = energy

        coord.update(kwargs)

        val = self.map.interp_by_coord(coord, **self._interp_kwargs)
        val = np.clip(val, 0, a_max=None)

        return u.Quantity(val, self.map.unit, copy=COPY_IF_NEEDED)

    def write(self, overwrite=False):
        """
        Write the map.

        Parameters
        ----------
        overwrite: bool, optional
            Overwrite existing file.
            Default is False, which will raise a warning if the template file exists already.
        """
        if self.filename is None:
            raise IOError("Missing filename")
        elif os.path.isfile(self.filename) and not overwrite:
            log.warning("Template file already exits, and overwrite is False")
        else:
            self.map.write(self.filename)

    @classmethod
    def from_dict(cls, data):
        data = data["spatial"]
        filename = data["filename"]
        m = Map.read(filename)
        model = cls(m, filename=filename)
        for idx, p in enumerate(model.parameters):
            par = p.to_dict()
            par.update(data["parameters"][idx])
            setattr(model, p.name, Parameter(**par))
        return model

    def to_dict(self, full_output=False):
        """Create dictionary for YAML serialisation."""
        data = super().to_dict(full_output)
        data["spatial"]["filename"] = self.filename
        data["spatial"]["unit"] = str(self.map.unit)
        return data


class PiecewiseNormSpatialModel(SpatialModel):
    """Piecewise spatial correction with a free normalization at each fixed nodes.

       For more information see :ref:`piecewise-norm-spatial`.

    Parameters
    ----------
    coord : `gammapy.maps.MapCoord`
        Flat coordinates list at which the model values are given (nodes).
    norms : `~numpy.ndarray` or list of `Parameter`
        Array with the initial norms of the model at energies ``energy``.
        Normalisation parameters are created for each value.
        Default is one at each node.
    interp : {"lin", "log"}
        Interpolation scaling. Default is "lin".
    """

    tag = ["PiecewiseNormSpatialModel", "piecewise-norm"]

    def __init__(self, coords, norms=None, interp="lin", **kwargs):
        self._coords = coords.copy()
        lon = Angle(coords.lon).wrap_at(0 * u.deg)
        self._wrap_angle = (lon.max() - lon.min()) / 2
        self._coords["lon"] = Angle(lon).wrap_at(self._wrap_angle)
        self._interp = interp

        if norms is None:
            norms = np.ones(coords.shape)

        if len(norms) != coords.shape[0]:
            raise ValueError("dimension mismatch")

        if len(norms) < 4:
            raise ValueError("Input arrays must contain at least 4 elements")

        if self.is_energy_dependent:
            raise ValueError("Energy dependent nodes are not supported")

        if not isinstance(norms[0], Parameter):
            parameters = Parameters(
                [Parameter(f"norm_{k}", norm) for k, norm in enumerate(norms)]
            )
        else:
            parameters = Parameters(norms)
        self.default_parameters = parameters
        super().__init__(**kwargs)

    @property
    def coords(self):
        """Energy nodes."""
        return self._coords

    @property
    def norms(self):
        """Norm values."""
        return u.Quantity([p.quantity for p in self.parameters])

    @property
    def is_energy_dependent(self):
        keys = self.coords._data.keys()
        return "energy" in keys or "energy_true" in keys

    def evaluate(self, lon, lat, energy=None, **norms):
        """Evaluate the model at given coordinates."""
        scale = interpolation_scale(scale=self._interp)
        v_nodes = scale(self.norms.value)
        coords = [value.value for value in self.coords._data.values()]
        # TODO: apply axes scaling in this loop
        coords = list(zip(*coords))
        lon = Angle(lon).wrap_at(0 * u.deg)
        lon = Angle(lon).wrap_at(self._wrap_angle)
        # by default rely on CloughTocher2DInterpolator
        # (Piecewise cubic, C1 smooth, curvature-minimizing interpolant)
        interpolated = griddata(coords, v_nodes, (lon, lat), method="cubic")
        return scale.inverse(interpolated) * self.norms.unit

    def evaluate_geom(self, geom):
        """Evaluate model on `~gammapy.maps.Geom`.

        Parameters
        ----------
        geom : `~gammapy.maps.WcsGeom`
            Map geometry.

        Returns
        -------
        map : `~gammapy.maps.Map`
            Map containing the value in each spatial bin.

        """
        coords = geom.get_coord(frame=self.frame, sparse=True)
        return self(coords.lon, coords.lat)

    def to_dict(self, full_output=False):
        data = super().to_dict(full_output=full_output)
        for key, value in self.coords._data.items():
            data["spatial"][key] = {
                "data": value.data.tolist(),
                "unit": str(value.unit),
            }
        return data

    @classmethod
    def from_dict(cls, data):
        """Create model from dictionary."""
        data = data["spatial"]
        lon = u.Quantity(data["lon"]["data"], data["lon"]["unit"])
        lat = u.Quantity(data["lat"]["data"], data["lat"]["unit"])
        coords = MapCoord.create((lon, lat))

        parameters = Parameters.from_dict(data["parameters"])
        return cls.from_parameters(parameters, coords=coords, frame=data["frame"])

    @classmethod
    def from_parameters(cls, parameters, **kwargs):
        """Create model from parameters."""
        return cls(norms=parameters, **kwargs)
