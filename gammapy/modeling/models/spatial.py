# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spatial models."""
import logging
import numpy as np
import scipy.integrate
import scipy.special
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.coordinates.angle_utilities import angular_separation, position_angle
from regions import (
    CircleAnnulusSkyRegion,
    EllipseSkyRegion,
    PointSkyRegion,
    PolygonSkyRegion,
)
from gammapy.maps import Map
from gammapy.modeling import Model, Parameter
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.scripts import make_path

log = logging.getLogger(__name__)


def compute_sigma_eff(lon_0, lat_0, lon, lat, phi, major_axis, e):
    """Effective radius, used for the evaluation of elongated models"""
    phi_0 = position_angle(lon_0, lat_0, lon, lat)
    d_phi = phi - phi_0
    minor_axis = Angle(major_axis * np.sqrt(1 - e ** 2))

    a2 = (major_axis * np.sin(d_phi)) ** 2
    b2 = (minor_axis * np.cos(d_phi)) ** 2
    denominator = np.sqrt(a2 + b2)
    sigma_eff = major_axis * minor_axis / denominator
    return minor_axis, sigma_eff


class SpatialModel(Model):
    """Spatial model base class."""

    def __init__(self, **kwargs):
        frame = kwargs.pop("frame", "icrs")
        super().__init__(**kwargs)
        if not hasattr(self, "frame"):
            self.frame = frame

    def __call__(self, lon, lat):
        """Call evaluate method"""
        kwargs = {par.name: par.quantity for par in self.parameters}
        return self.evaluate(lon, lat, **kwargs)

    @property
    def position(self):
        """Spatial model center position"""
        lon = self.lon_0.quantity
        lat = self.lat_0.quantity
        return SkyCoord(lon, lat, frame=self.frame)

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
        """Get 95% containment position error as (`~regions.EllipseSkyRegion`)"""
        if self.parameters.covariance is None:
            return EllipseSkyRegion(
                center=self.position,
                height=np.nan * u.deg,
                width=np.nan * u.deg,
                angle=np.nan * u.deg,
            )
        pars = self.parameters
        sub_covar = pars.get_subcovariance(["lon_0", "lat_0"])
        cos_lat = np.cos(self.lat_0.quantity.to_value("rad"))
        sub_covar[0, 0] *= cos_lat ** 2.0
        sub_covar[0, 1] *= cos_lat
        sub_covar[1, 0] *= cos_lat
        eig_vals, eig_vecs = np.linalg.eig(sub_covar)
        lon_err, lat_err = np.sqrt(eig_vals)
        y_vec = eig_vecs[:, 0]
        phi = (np.arctan2(y_vec[1], y_vec[0]) * u.rad).to("deg") + self.phi_0
        err = np.sort([lon_err, lat_err])
        scale_r95 = Gauss2DPDF().containment_radius(0.95)
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
        """Evaluate model on `~gammapy.maps.Geom`."""
        coords = geom.get_coord(coordsys=self.frame)
        return self(coords.lon, coords.lat)

    def to_dict(self):
        """Create dict for YAML serilisation"""
        data = super().to_dict()
        data["frame"] = self.frame
        data["parameters"] = data.pop("parameters")
        return data


class PointSpatialModel(SpatialModel):
    r"""Point Source.

    .. math:: \phi(lon, lat) = \delta{(lon - lon_0, lat - lat_0)}

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position
    frame : {"icrs", "galactic"}
        Center position coordinate frame
    """

    tag = "PointSpatialModel"
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)

    @property
    def evaluation_radius(self):
        """Evaluation radius (`~astropy.coordinates.Angle`).

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

    def evaluate_geom(self, geom):
        """Evaluate model on `~gammapy.maps.Geom`."""
        x, y = geom.get_pix()
        x0, y0 = self.position.to_pixel(geom.wcs)
        w = self._grid_weights(x, y, x0, y0)
        return w / geom.solid_angle()

    def to_region(self, **kwargs):
        """Model outline (`~regions.PointSkyRegion`)."""
        return PointSkyRegion(center=self.position, **kwargs)


class GaussianSpatialModel(SpatialModel):
    r"""Two-dimensional Gaussian model.

    By default, the Gaussian is symmetric:

    .. math::
        \phi(\text{lon}, \text{lat}) = N \times \exp\left\{-\frac{1}{2}
            \frac{1-\cos \theta}{1-\cos \sigma}\right\}\,,

    where :math:`\theta` is the sky separation to the model center. In this case, the
    Gaussian is normalized to 1 on the sphere:

    .. math::
        N = \frac{1}{4\pi a\left[1-\exp(-1/a)\right]}\,,\,\,\,\,
        a = 1-\cos \sigma\,.

    In the limit of small :math:`\theta` and :math:`\sigma`, this definition
    reduces to the usual form:

    .. math::
        \phi(\text{lon}, \text{lat}) = \frac{1}{2\pi\sigma^2} \exp{\left(-\frac{1}{2}
            \frac{\theta^2}{\sigma^2}\right)}\,.

    In case an eccentricity (:math:`e`) and rotation angle (:math:`\phi`) are passed,
    then the model is an elongated Gaussian, whose evaluation is performed as in the symmetric case
    but using the effective radius of the Gaussian:

    .. math::
        \sigma_{eff}(\text{lon}, \text{lat}) = \sqrt{
            (\sigma_M \sin(\Delta \phi))^2 +
            (\sigma_m \cos(\Delta \phi))^2
        }.

    Here, :math:`\sigma_M` (:math:`\sigma_m`) is the major (minor) semiaxis of the Gaussian, and
    :math:`\Delta \phi` is the difference between `phi`, the position angle of the Gaussian, and the
    position angle of the evaluation point.

    **Caveat:** For the asymmetric Gaussian, the model is normalized to 1 on the plane, i.e. in small angle
    approximation: :math:`N = 1/(2 \pi \sigma_M \sigma_m)`. This means that for huge elongated Gaussians on the sky
    this model is not correctly normalized. However, this approximation is perfectly acceptable for the more
    common case of models with modest dimensions: indeed, the error introduced by normalizing on the plane
    rather than on the sphere is below 0.1\% for Gaussians with radii smaller than ~ 5 deg.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position
    sigma : `~astropy.coordinates.Angle`
        Length of the major semiaxis of the Gaussian, in angular units.
    e : `float`
        Eccentricity of the Gaussian (:math:`0< e< 1`).
    phi : `~astropy.coordinates.Angle`
        Rotation angle :math:`\phi`: of the major semiaxis.
        Increases counter-clockwise from the North direction.
    frame : {"icrs", "galactic"}
        Center position coordinate frame
    """

    tag = "GaussianSpatialModel"

    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)
    sigma = Parameter("sigma", "1 deg", min=0)
    e = Parameter("e", 0, min=0, max=1, frozen=True)
    phi = Parameter("phi", "0 deg", frozen=True)

    @property
    def evaluation_radius(self):
        r"""Evaluation radius (`~astropy.coordinates.Angle`).

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
        return u.Quantity(norm * np.exp(exponent).value, "sr-1", copy=False)

    def to_region(self, **kwargs):
        """Model outline (`~regions.EllipseSkyRegion`)."""
        minor_axis = Angle(self.sigma.quantity * np.sqrt(1 - self.e.quantity ** 2))
        return EllipseSkyRegion(
            center=self.position,
            height=2 * self.sigma.quantity,
            width=2 * minor_axis,
            angle=self.phi.quantity,
            **kwargs
        )


class DiskSpatialModel(SpatialModel):
    r"""Constant disk model.

    By default, the model is symmetric, i.e. a disk:

    .. math::
        \phi(lon, lat) = \frac{1}{2 \pi (1 - \cos{r_0}) } \cdot
                \begin{cases}
                    1 & \text{for } \theta \leq r_0 \\
                    0 & \text{for } \theta > r_0
                \end{cases}

    where :math:`\theta` is the sky separation. To improve fit convergence of the
    model, the sharp edges is smoothed using `~scipy.special.erf`.

    In case an eccentricity (`e`) and rotation angle (:math:`\phi`) are passed,
    then the model is an elongated disk (i.e. an ellipse), with a major semiaxis of length :math:`r_0`
    and position angle :math:`\phi` (increaing counter-clockwise from the North direction).

    The model is defined on the celestial sphere, with a normalization defined by:

    .. math::
        \int_{4\pi}\phi(\text{lon}, \text{lat}) \,d\Omega = 1\,.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position
    r_0 : `~astropy.coordinates.Angle`
        :math:`a`: length of the major semiaxis, in angular units.
    e : `float`
        Eccentricity of the ellipse (:math:`0< e< 1`).
    phi : `~astropy.coordinates.Angle`
        Rotation angle :math:`\phi`: of the major semiaxis.
        Increases counter-clockwise from the North direction.
    edge : `~astropy.coordinates.Angle`
        Width of the edge. The width is defined as the range within the
        smooth edges of the model drops from 95% to 5% of its amplitude.
    frame : {"icrs", "galactic"}
        Center position coordinate frame
    """

    tag = "DiskSpatialModel"
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)
    r_0 = Parameter("r_0", "1 deg", min=0)
    e = Parameter("e", 0, min=0, max=1, frozen=True)
    phi = Parameter("phi", "0 deg", frozen=True)
    edge = Parameter("edge", "0.01 deg", frozen=True, min=0.01)

    @property
    def evaluation_radius(self):
        """Evaluation radius (`~astropy.coordinates.Angle`).

        Set to the length of the semi-major axis.
        """
        return self.parameters["r_0"].quantity

    @staticmethod
    def _evaluate_norm_factor(r_0, e):
        """Compute the normalization factor."""
        semi_minor = r_0 * np.sqrt(1 - e ** 2)

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
    def evaluate(lon, lat, lon_0, lat_0, r_0, e, phi, edge):
        """Evaluate model."""
        sep = angular_separation(lon, lat, lon_0, lat_0)

        if e == 0:
            sigma_eff = r_0
        else:
            sigma_eff = compute_sigma_eff(lon_0, lat_0, lon, lat, phi, r_0, e)[1]

        norm = DiskSpatialModel._evaluate_norm_factor(r_0, e)

        in_ellipse = DiskSpatialModel._evaluate_smooth_edge(sep - sigma_eff, edge)
        return u.Quantity(norm * in_ellipse, "sr-1", copy=False)

    def to_region(self, **kwargs):
        """Model outline (`~regions.EllipseSkyRegion`)."""
        minor_axis = Angle(self.r_0.quantity * np.sqrt(1 - self.e.quantity ** 2))
        return EllipseSkyRegion(
            center=self.position,
            height=2 * self.r_0.quantity,
            width=2 * minor_axis,
            angle=self.phi.quantity,
            **kwargs
        )


class ShellSpatialModel(SpatialModel):
    r"""Shell model.

    .. math::
        \phi(lon, lat) = \frac{3}{2 \pi (r_{out}^3 - r_{in}^3)} \cdot
                \begin{cases}
                    \sqrt{r_{out}^2 - \theta^2} - \sqrt{r_{in}^2 - \theta^2} &
                                 \text{for } \theta \lt r_{in} \\
                    \sqrt{r_{out}^2 - \theta^2} &
                                 \text{for } r_{in} \leq \theta \lt r_{out} \\
                    0 & \text{for } \theta > r_{out}
                \end{cases}

    where :math:`\theta` is the sky separation and :math:`r_{\text{out}} = r_{\text{in}}` + width

    Note that the normalization is a small angle approximation,
    although that approximation is still very good even for 10 deg radius shells.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position
    radius : `~astropy.coordinates.Angle`
        Inner radius, :math:`r_{in}`
    width : `~astropy.coordinates.Angle`
        Shell width
    frame : {"icrs", "galactic"}
        Center position coordinate frame
    """

    tag = "ShellSpatialModel"
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)
    radius = Parameter("radius", "1 deg")
    width = Parameter("width", "1 deg")

    @property
    def evaluation_radius(self):
        r"""Evaluation radius (`~astropy.coordinates.Angle`).

        Set to :math:`r_\text{out}`.
        """
        return self.parameters["radius"].quantity + self.parameters["width"].quantity

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, radius, width):
        """Evaluate model."""
        sep = angular_separation(lon, lat, lon_0, lat_0)
        radius_out = radius + width

        norm = 3 / (2 * np.pi * (radius_out ** 3 - radius ** 3))

        with np.errstate(invalid="ignore"):
            # np.where and np.select do not work with quantities, so we use the
            # workaround with indexing
            value = np.sqrt(radius_out ** 2 - sep ** 2)
            mask = [sep < radius]
            value[mask] = (value - np.sqrt(radius ** 2 - sep ** 2))[mask]
            value[sep > radius_out] = 0

        return norm * value

    def to_region(self, **kwargs):
        """Model outline (`~regions.CircleAnnulusSkyRegion`)."""
        return CircleAnnulusSkyRegion(
            center=self.position,
            inner_radius=self.radius.quantity,
            outer_radius=self.radius.quantity + self.width.quantity,
            **kwargs
        )


class ConstantSpatialModel(SpatialModel):
    """Spatially constant (isotropic) spatial model.

    Parameters
    ----------
    value : `~astropy.units.Quantity`
        Value
    """

    tag = "ConstantSpatialModel"
    value = Parameter("value", "1 sr-1", frozen=True)

    frame = "icrs"
    evaluation_radius = None

    def to_dict(self):
        """Create dict for YAML serilisation"""
        # redefined to ignore frame attribute from parent class
        data = super().to_dict()
        data.pop("frame")
        data["parameters"] = data.pop("parameters")
        return data

    @staticmethod
    def evaluate(lon, lat, value):
        """Evaluate model."""
        return value

    @staticmethod
    def to_region(**kwargs):
        """Model outline (`~regions.EllipseSkyRegion`)."""
        return EllipseSkyRegion(
            center=SkyCoord(np.nan * u.deg, np.nan * u.deg),
            height=np.nan * u.deg,
            width=np.nan * u.deg,
            angle=np.nan * u.deg,
            **kwargs
        )


class TemplateSpatialModel(SpatialModel):
    """Spatial sky map template model (2D).

    This is for a 2D image. Use `~gammapy.modeling.models.SkyDiffuseCube` for 3D cubes with
    an energy axis.

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Map template
    norm : float
        Norm parameter (multiplied with map values)
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    normalize : bool
        Normalize the input map so that it integrates to unity.
    interp_kwargs : dict
        Interpolation keyword arguments passed to `gammapy.maps.Map.interp_by_coord`.
        Default arguments are {'interp': 'linear', 'fill_value': 0}.
    """

    tag = "TemplateSpatialModel"
    norm = Parameter("norm", 1)

    def __init__(
        self,
        map,
        norm=norm.quantity,
        meta=None,
        normalize=True,
        interp_kwargs=None,
        filename=None,
    ):
        if (map.data < 0).any():
            log.warning("Diffuse map has negative values. Check and fix this!")

        if filename is not None:
            filename = str(make_path(filename))

        self.map = map
        self.normalize = normalize
        if normalize:
            # Normalize the diffuse map model so that it integrates to unity."""
            data = self.map.data / self.map.data.sum()
            data /= self.map.geom.solid_angle().to_value("sr")
            self.map = self.map.copy(data=data, unit="sr-1")

        self.meta = dict() if meta is None else meta
        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        interp_kwargs.setdefault("interp", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        self._interp_kwargs = interp_kwargs
        self.filename = filename
        super().__init__(norm=norm)

    @property
    def evaluation_radius(self):
        """Evaluation radius (`~astropy.coordinates.Angle`).

        Set to half of the maximal dimension of the map.
        """
        return np.max(self.map.geom.width) / 2.0

    @classmethod
    def read(cls, filename, normalize=True, **kwargs):
        """Read spatial template model from FITS image.

        The default unit used if none is found in the file is ``sr-1``.

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
        if m.unit == "":
            m.unit = "sr-1"
        return cls(m, normalize=normalize, filename=filename)

    def evaluate(self, lon, lat, norm):
        """Evaluate model."""
        coord = {"lon": lon.to_value("deg"), "lat": lat.to_value("deg")}
        val = self.map.interp_by_coord(coord, **self._interp_kwargs)
        return u.Quantity(norm.value * val, self.map.unit, copy=False)

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return self.map.geom.center_skydir

    @property
    def frame(self):
        return self.position.frame.name

    @classmethod
    def from_dict(cls, data):
        model = cls.read(data["filename"], normalize=data.get("normalize", True))
        model._update_from_dict(data)
        return model

    def to_dict(self):
        """Create dict for YAML serilisation"""
        data = super().to_dict()
        data["filename"] = self.filename
        data["normalize"] = self.normalize
        return data

    def to_region(self, **kwargs):
        """Model outline (`~regions.PolygonSkyRegion`)."""
        footprint = self.map.geom.wcs.calc_footprint()
        return PolygonSkyRegion(
            vertices=SkyCoord(footprint, unit="deg", frame=self.frame, **kwargs)
        )
