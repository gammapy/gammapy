# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import Angle, Longitude, Latitude, SkyCoord
from ...utils.fitting import Parameter, Parameters, Model
from ...maps import Map
from scipy.integrate import quad

__all__ = [
    "SkySpatialModel",
    "SkyPointSource",
    "SkyGaussian",
    "SkyDisk",
    "SkyEllipse",
    "SkyShell",
    "SkyDiffuseConstant",
    "SkyDiffuseMap",
]


log = logging.getLogger(__name__)


class SkySpatialModel(Model):
    """Sky spatial model base class."""
    def __call__(self, lon, lat):
        """Call evaluate method"""
        kwargs = dict()
        for par in self.parameters.parameters:
            kwargs[par.name] = par.quantity

        return self.evaluate(lon, lat, **kwargs)

    @property
    def position(self):
        """Spatial model center position"""
        try:
            # TODO: this relies on hard-coded parameter names, which is not the ideal solution
            lon = self.parameters["lon_0"].quantity
            lat = self.parameters["lat_0"].quantity
            return SkyCoord(lon, lat, frame=self.frame)
        except IndexError:
            raise ValueError("Model does not have a defined center position")


class SkyPointSource(SkySpatialModel):
    r"""Point Source.

    .. math::

        \phi(lon, lat) = \delta{(lon - lon_0, lat - lat_0)}


    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`lon_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`lat_0`
    frame : {"galactic", "icrs"}
        Coordinate frame of `lon_0` and `lat_0`.
    """

    def __init__(self, lon_0, lat_0, frame="galactic"):
        self.frame = frame
        self.parameters = Parameters(
            [Parameter("lon_0", Longitude(lon_0).wrap_at("180d"), min=-180, max=180), Parameter("lat_0", Latitude(lat_0), min=-90, max=90)]
        )

    @property
    def evaluation_radius(self):
        """Returns the effective radius of the sky region where the model evaluates to non-zero.
        For a Gaussian source, we fix it to :math:`0`.

        Returns
        -------
        radius : `~astropy.units.Quantity`
            Radius
        """
        radius = 0 * u.deg
        return radius

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0):
        """Evaluate the model (static function)."""

        wrapval = lon_0 + 180 * u.deg
        lon = Angle(lon).wrap_at(wrapval)

        _, grad_lon = np.gradient(lon)
        grad_lat, _ = np.gradient(lat)
        lon_diff = np.abs((lon - lon_0) / grad_lon)
        lat_diff = np.abs((lat - lat_0) / grad_lat)

        lon_val = np.select([lon_diff < 1], [1 - lon_diff], 0) / np.abs(grad_lon)
        lat_val = np.select([lat_diff < 1], [1 - lat_diff], 0) / np.abs(grad_lat)
        return lon_val * lat_val


class SkyGaussian(SkySpatialModel):
    r"""Two-dimensional symmetric Gaussian model

    .. math::

        \phi(\text{lon}, \text{lat}) = N \times \text{exp}\left\{-\frac{1}{2}
            \frac{1-\text{cos}\theta}{1-\text{cos}\sigma}\right\}\,,

    where :math:`\theta` is the angular separation between the center of the Gaussian and the evaluation point.
    This angle is calculated on the celestial sphere using the function `angular.separation` defined in
    `astropy.coordinates.angle_utilities`. The Gaussian is normalized to 1 on
    the sphere:

    .. math::

        N = \frac{1}{4\pi a\left[1-\text{exp}(-1/a)\right]}\,,\,\,\,\,
        a = 1-\text{cos}\sigma\,.

    The normalization factor is in units of :math:`\text{sr}^{-1}`.
    In the limit of small :math:`\theta` and :math:`\sigma`, this definition reduces to the usual form:

    .. math::

        \phi(\text{lon}, \text{lat}) = \frac{1}{2\pi\sigma^2} \exp{\left(-\frac{1}{2}
            \frac{\theta^2}{\sigma^2}\right)}


    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`\text{lon}_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`\text{lat}_0`
    sigma : `~astropy.coordinates.Angle`
        :math:`\sigma`
    frame : {"galactic", "icrs"}
        Coordinate frame of `lon_0` and `lat_0`.
    """

    def __init__(self, lon_0, lat_0, sigma, frame="galactic"):
        self.frame = frame
        self.parameters = Parameters(
            [
                Parameter("lon_0", Longitude(lon_0).wrap_at("180d"), min=-180, max=180),
                Parameter("lat_0", Latitude(lat_0), min=-90, max=90),
                Parameter("sigma", Angle(sigma), min=0),
            ]
        )

    @property
    def evaluation_radius(self):
        r"""Returns the effective radius of the sky region where the model evaluates to non-zero.
        For a Gaussian source, we fix it to :math:`5\sigma`.

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
           Radius in angular units

        """
        radius = 5 * self.parameters["sigma"].quantity
        return radius

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, sigma):
        """Evaluate the model (static function)."""
        sep = angular_separation(lon, lat, lon_0, lat_0)
        a = 1.0 - np.cos(sigma)
        norm = 1 / (4 * np.pi * a * (1.0 - np.exp(-1.0 / a)))
        exponent = -0.5 * ((1 - np.cos(sep)) / a)
        return u.Quantity(norm.value * np.exp(exponent).value, "sr-1", copy=False)


class SkyDisk(SkySpatialModel):
    r"""Constant radial disk model.

    .. math::

        \phi(lon, lat) = \frac{1}{2 \pi (1 - \cos{r_0}) } \cdot
                \begin{cases}
                    1 & \text{for } \theta \leq r_0 \\
                    0 & \text{for } \theta > r_0
                \end{cases}

    where :math:`\theta` is the sky separation

    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`lon_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`lat_0`
    r_0 : `~astropy.coordinates.Angle`
        :math:`r_0`
    frame : {"galactic", "icrs"}
        Coordinate frame of `lon_0` and `lat_0`.
    """

    def __init__(self, lon_0, lat_0, r_0, frame="galactic"):
        self.frame = frame
        self.parameters = Parameters(
            [
                Parameter("lon_0", Longitude(lon_0).wrap_at("180d"), min=-180, max=180),
                Parameter("lat_0", Latitude(lat_0), min=-90, max=90),
                Parameter("r_0", Angle(r_0)),
            ]
        )

    @property
    def evaluation_radius(self):
        r"""Returns the effective radius of the sky region where the model evaluates to non-zero.
        For a Disk source, we fix it to :math:`r_0`.

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
            Radius in angular units

        """
        radius = self.parameters["r_0"].quantity
        return radius

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, r_0):
        """Evaluate the model (static function)."""
        sep = angular_separation(lon, lat, lon_0, lat_0)

        # Surface area of a spherical cap, see https://en.wikipedia.org/wiki/Spherical_cap
        norm = 1.0 / (2 * np.pi * (1 - np.cos(r_0)))
        return u.Quantity(norm.value * (sep <= r_0), "sr-1", copy=False)


class SkyEllipse(SkySpatialModel):
    r"""Constant elliptical model

    .. math::
       \phi(\text{lon}, \text{lat}) =
                \begin{cases}
                    N & \text{for }  \,\,\,dist(F_1,P)+dist(F_2,P)\leq 2 a \\
                    0 & \text{otherwise }\,,
                \end{cases}



    where :math:`F_1` and :math:`F_2` represent the foci of the ellipse,
    :math:`P` is a generic point of coordinates :math:`(\text{lon}, \text{lat})`,
    :math:`a` is the major semiaxis of the ellipse and N is the model's
    normalization, in units of :math:`\text{sr}^{-1}`.

    The model is defined on the celestial sphere, by computing angles and distances with the functions defined in
    `astropy.coordinates.angle_utilities`, and setting the normalization such that:

    .. math::
       \int_{4\pi}\phi(\text{lon}, \text{lat}) \,d\Omega = 1\,.


    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`\text{lon}_0`: `lon` coordinate for the center of the ellipse.
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`\text{lat}_0`: `lat` coordinate for the center of the ellipse.
    semi_major : `~astropy.coordinates.Angle`
        :math:`a`: length of the major semiaxis, in angular units.
    e : `float`
        Eccentricity of the ellipse (:math:`0< e< 1`).
    theta : `~astropy.coordinates.Angle`
        :math:`\theta`:
        Rotation angle of the major semiaxis.  The rotation angle increases clockwise
        (i.e., East of North) from the positive `lon` axis.
    frame : {"galactic", "icrs"}
        Coordinate frame of `lon_0` and `lat_0`.


    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        from gammapy.image.models.core import SkyEllipse
        from gammapy.maps import Map, WcsGeom

        model = SkyEllipse("2 deg", "2 deg", "1 deg", 0.8, "30 deg")

        m_geom = WcsGeom.create(binsz=0.01, width=(3, 3), skydir=(2, 2), coordsys="GAL", proj="AIT")
        coords = m_geom.get_coord()
        lon = coords.lon * u.deg
        lat = coords.lat * u.deg
        vals = model(lon, lat)
        skymap = Map.from_geom(m_geom, data=vals.value)

        _, ax, _ = skymap.smooth("0.05 deg").plot()
        transform = ax.get_transform('galactic')
        ax.scatter(2, 2, transform=transform, s=20, edgecolor='red', facecolor='red')
        ax.text(2.0, 1.8, r"$(l_0, b_0)$", transform=transform, ha="center")
        ax.plot([2, 2 + np.cos(np.pi / 6)], [2, 2 + np.sin(np.pi / 6)], color="r", transform=transform)
        ax.hlines(y=2, color='r', linestyle='--', transform=transform, xmin=0, xmax=5)
        ax.text(2.5, 2.06, r"$\theta$", transform=transform);

        plt.show()
    """

    def __init__(self, lon_0, lat_0, semi_major, e, theta, frame="galactic"):
        try:
            from astropy.coordinates.angle_utilities import offset_by

            self._offset_by = offset_by
        except ImportError:
            raise ImportError("The SkyEllipse model requires astropy>=3.1")

        self.frame = frame
        self.parameters = Parameters(
            [
                Parameter("lon_0", Longitude(lon_0).wrap_at("180d"), min=-180, max=180),
                Parameter("lat_0", Latitude(lat_0), min=-90, max=90),
                Parameter("semi_major", Angle(semi_major)),
                Parameter("e", e, min=0, max=1),
                Parameter("theta", Angle(theta)),
            ]
        )

    @property
    def evaluation_radius(self):
        r"""Returns the effective radius of the sky region where the model evaluates to non-zero.
        For an elliptical source, we fix it to the length of the semi-major axis.

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
            Radius in angular units

        """
        radius = self.parameters["semi_major"].quantity
        return radius

    def compute_norm(semi_major, e):
        """Compute the normalization factor."""
        semi_minor = semi_major * np.sqrt(1 - e ** 2)

        def integral_fcn(x, a, b):
            A = 1 / np.sin(a) ** 2
            B = 1 / np.sin(b) ** 2
            C = A - B
            cs2 = np.cos(x) ** 2

            return 1 - np.sqrt(1 - 1 / (B + C * cs2))

        return (
            2 * quad(lambda x: integral_fcn(x, semi_major, semi_minor), 0, np.pi)[0]
        ) ** -1

    def evaluate(self, lon, lat, lon_0, lat_0, semi_major, e, theta):
        """Evaluate the model (static function)."""

        # find the foci of the ellipse
        c = semi_major * e
        lon_1, lat_1 = self._offset_by(lon_0, lat_0, 90 * u.deg - theta, c)
        lon_2, lat_2 = self._offset_by(lon_0, lat_0, 270 * u.deg - theta, c)

        sep_1 = angular_separation(lon, lat, lon_1, lat_1)
        sep_2 = angular_separation(lon, lat, lon_2, lat_2)
        in_ellipse = sep_1 + sep_2 <= 2 * semi_major

        norm = SkyEllipse.compute_norm(semi_major, e)
        return u.Quantity(norm * in_ellipse, "sr-1", copy=False)


class SkyShell(SkySpatialModel):
    r"""Shell model

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
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`lon_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`lat_0`
    radius : `~astropy.coordinates.Angle`
        Inner radius, :math:`r_{in}`
    width : `~astropy.coordinates.Angle`
        Shell width
    frame : {"galactic", "icrs"}
        Coordinate frame of `lon_0` and `lat_0`.
    """

    def __init__(self, lon_0, lat_0, radius, width, frame="galactic"):
        self.frame = frame
        self.parameters = Parameters(
            [
                Parameter("lon_0", Longitude(lon_0).wrap_at("180d"), min=-180, max=180),
                Parameter("lat_0", Latitude(lat_0), min=-90, max=90),
                Parameter("radius", Angle(radius)),
                Parameter("width", Angle(width)),
            ]
        )

    @property
    def evaluation_radius(self):
        r"""Returns the effective radius of the sky region where the model evaluates to non-zero.
        For a Shell source, we fix it to :math:`r_\text{out}`.

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
            Radius in angular units

        """
        radius = self.parameters["radius"].quantity + self.parameters["width"].quantity
        return radius

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, radius, width):
        """Evaluate the model (static function)."""
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


class SkyDiffuseConstant(SkySpatialModel):
    """Spatially constant (isotropic) spatial model.

    Parameters
    ----------
    value : `~astropy.units.Quantity`
        Value
    """
    frame = None
    def __init__(self, value=1):
        self.parameters = Parameters([Parameter("value", value)])

    @property
    def evaluation_radius(self):
        r"""Returns the effective radius of the sky region where the model evaluates to non-zero.
        For a constant diffuse model, we fix it to None.

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
             None

        """
        return None

    @staticmethod
    def evaluate(lon, lat, value):
        return value


class SkyDiffuseMap(SkySpatialModel):
    """Spatial sky map template model (2D).

    This is for a 2D image. Use `~gammapy.cube.SkyDiffuseCube` for 3D cubes with
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
        Interpolation keyword arguments passed to `Map.interp_by_coord()`.
        Default arguments are {'interp': 'linear', 'fill_value': 0}.
    """

    def __init__(self, map, norm=1, meta=None, normalize=True, interp_kwargs=None):
        if (map.data < 0).any():
            log.warn(
                "Map template contains negative values, please check the"
                " data and fix if needed."
            )

        self.map = map

        if normalize:
            self.normalize()

        self.parameters = Parameters([Parameter("norm", norm)])
        self.meta = dict() if meta is None else meta

        interp_kwargs = {} if interp_kwargs is None else interp_kwargs
        interp_kwargs.setdefault("interp", "linear")
        interp_kwargs.setdefault("fill_value", 0)
        self._interp_kwargs = interp_kwargs

    @property
    def evaluation_radius(self):
        r"""Returns the effective radius of the sky region where the model evaluates to non-zero.
        For a DiffuseMap source, we fix it to half of the maximal dimension of the map.

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
            Radius in angular units.

        """
        radius = np.max(self.map.geom.width) / 2.
        return radius

    def normalize(self):
        """Normalize the diffuse map model so that it integrates to unity."""
        data = self.map.data / self.map.data.sum()
        data /= self.map.geom.solid_angle().to_value("sr")
        self.map = self.map.copy(data=data, unit="sr-1")

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
        return cls(m, normalize=normalize)

    def evaluate(self, lon, lat, norm):
        """Evaluate model."""
        coord = {"lon": lon.to_value("deg"), "lat": lat.to_value("deg")}
        val = self.map.interp_by_coord(coord, **self._interp_kwargs)
        return u.Quantity(norm.value * val, self.map.unit, copy=False)

    @property
    def position(self):
        """`~astropy.coordinates.SkyCoord`"""
        return self.map.geom.center_skydir
