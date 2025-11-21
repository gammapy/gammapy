# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    AltAz,
    Angle,
    BaseCoordinateFrame,
    CoordinateAttribute,
    DynamicMatrixTransform,
    EarthLocationAttribute,
    FunctionTransform,
    ICRS,
    RepresentationMapping,
    SkyCoord,
    SkyOffsetFrame,
    TimeAttribute,
    UnitSphericalRepresentation,
    frame_transform_graph,
)
from astropy.coordinates.matrix_utilities import matrix_transpose, rotation_matrix

__all__ = ["FoVAltAzFrame", "FoVICRSFrame", "fov_to_sky", "sky_to_fov"]

reflect_lon_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])


class FoVAltAzFrame(BaseCoordinateFrame):
    """
    FoV coordinate frame. Centered on `origin` and aligned on AltAz frame at `location` and `obstime`.

    Longitudes are reversed.

    Attributes
    ----------
    origin: `~astropy.coordinates.AltAz`
        Origin of this frame as an Altaz coordinate
    obstime: `~astropy.time.Time`
        Observation time
    location: `~astropy.coordinates.EarthLocation`
        Location of the telescope/instrument/observatory

    Examples
    --------
    .. testcode::

        import astropy.units as u
        from astropy.time import Time
        from astropy.coordinates import AltAz, SkyCoord
        from gammapy.utils.observers import observatory_locations
        from gammapy.utils.coordinates import FoVAltAzFrame

        location = observatory_locations.get("ctao_north")
        obstime = Time("2025-01-01T00:00:00")
        origin = AltAz(alt=80*u.deg, az=172*u.deg, location=location, obstime=obstime)

        fov_frame = FoVAltAzFrame(origin=origin, location=location, obstime=obstime)

        crab= SkyCoord(83.63333333, 22.01444444, unit="deg", frame="icrs")
        crab.transform_to(fov_frame)

    """

    frame_specific_representation_info = {
        UnitSphericalRepresentation: [
            RepresentationMapping("lon", "fov_lon"),
            RepresentationMapping("lat", "fov_lat"),
        ]
    }
    default_representation = UnitSphericalRepresentation

    origin = CoordinateAttribute(default=None, frame=AltAz)

    obstime = TimeAttribute(default=None)
    location = EarthLocationAttribute(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # make sure telescope coordinate is in range [-180°, 180°]
        if isinstance(self._data, UnitSphericalRepresentation):
            self._data.lon.wrap_angle = Angle(180, unit=u.deg)


@frame_transform_graph.transform(FunctionTransform, FoVAltAzFrame, FoVAltAzFrame)
def fov_to_fov(from_fov_altaz_coord, to_fov_altaz_frame):
    """Transform between two `FoVFrame`."""
    intermediate_from = from_fov_altaz_coord.transform_to(from_fov_altaz_coord.origin)
    intermediate_to = intermediate_from.transform_to(to_fov_altaz_frame.origin)
    return intermediate_to.transform_to(to_fov_altaz_frame)


@frame_transform_graph.transform(DynamicMatrixTransform, AltAz, FoVAltAzFrame)
def altaz_to_fov_altaz(altaz_coord, fov_altaz_frame):
    """Convert a reference coordinate to a sky offset frame."""
    # Define rotation matrices along the position angle vector, and
    # relative to the origin.
    origin = fov_altaz_frame.origin.represent_as(UnitSphericalRepresentation)
    mat1 = rotation_matrix(-origin.lat, "y")
    mat2 = rotation_matrix(origin.lon, "z")

    return reflect_lon_matrix @ mat1 @ mat2


@frame_transform_graph.transform(DynamicMatrixTransform, FoVAltAzFrame, AltAz)
def fov_to_altaz(fov_altaz_coord, altaz_frame):
    """Convert an sky offset frame coordinate to the reference frame."""
    # use the forward transform, but just invert it
    mat = altaz_to_fov_altaz(altaz_frame, fov_altaz_coord)
    return matrix_transpose(mat)


class FoVICRSFrame(BaseCoordinateFrame):
    """
    FoV coordinate frame aligned on ICRS frame. Centered on `origin` an ICRS coordinate and aligned on ICRS frame.

    Longitudes are reversed.

    Attributes
    ----------
    origin: `~astropy.coordinates.ICRS`
        Origin of this frame as an ICRS coordinate

    Examples
    --------
    .. testcode::

        import astropy.units as u
        from astropy.time import Time
        from astropy.coordinates import SkyCoord
        from gammapy.utils.observers import observatory_locations
        from gammapy.utils.coordinates import FoVICRSFrame

        location = observatory_locations.get("ctao_north")
        obstime = Time("2025-01-01T00:00:00")
        origin = SkyCoord(85.63333333, 20.01444444, unit="deg", frame="icrs")

        fov_frame = FoVICRSFrame(origin=origin)

        crab = SkyCoord(83.63333333, 22.01444444, unit="deg", frame="icrs")
        crab.transform_to(fov_frame)
    """

    frame_specific_representation_info = {
        UnitSphericalRepresentation: [
            RepresentationMapping("lon", "fov_lon"),
            RepresentationMapping("lat", "fov_lat"),
        ]
    }
    default_representation = UnitSphericalRepresentation

    origin = CoordinateAttribute(default=None, frame=ICRS)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # make sure telescope coordinate is in range [-180°, 180°]
        if isinstance(self._data, UnitSphericalRepresentation):
            self._data.lon.wrap_angle = Angle(180, unit=u.deg)


@frame_transform_graph.transform(FunctionTransform, FoVICRSFrame, FoVICRSFrame)
def fov_icrs_to_fov_icrs(from_fov_icrs_coord, to_fov_icrs_frame):
    """Transform between two `FoVFrame`."""
    intermediate_from = from_fov_icrs_coord.transform_to(from_fov_icrs_coord.origin)
    intermediate_to = intermediate_from.transform_to(to_fov_icrs_frame.origin)
    return intermediate_to.transform_to(to_fov_icrs_frame)


@frame_transform_graph.transform(DynamicMatrixTransform, ICRS, FoVICRSFrame)
def icrs_to_fov_icrs(icrs_coord, fov_icrs_frame):
    """Convert a reference coordinate to a sky offset frame."""
    # Define rotation matrices along the position angle vector, and
    # relative to the origin.
    origin = fov_icrs_frame.origin.represent_as(UnitSphericalRepresentation)
    mat1 = rotation_matrix(-origin.lat, "y")
    mat2 = rotation_matrix(origin.lon, "z")

    return reflect_lon_matrix @ mat1 @ mat2


@frame_transform_graph.transform(DynamicMatrixTransform, FoVICRSFrame, ICRS)
def fov_icrs_to_icrs(fov_icrs_coord, icrs_frame):
    """Convert an sky offset frame coordinate to the reference frame."""
    # use the forward transform, but just invert it
    mat = icrs_to_fov_icrs(icrs_frame, fov_icrs_coord)
    return matrix_transpose(mat)


def fov_to_sky(lon, lat, lon_pnt, lat_pnt):
    """Transform field-of-view coordinates to sky coordinates.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Field-of-view coordinate to be transformed.
    lon_pnt, lat_pnt : `~astropy.units.Quantity`
        Coordinate specifying the pointing position.
        (i.e. the center of the field of view.)

    Returns
    -------
    lon_t, lat_t : `~astropy.units.Quantity`
        Transformed sky coordinate.
    """
    # Create a frame that is centered on the pointing position
    center = SkyCoord(lon_pnt, lat_pnt)
    fov_frame = SkyOffsetFrame(origin=center)

    # Define coordinate to be transformed.
    # Need to switch the sign of the longitude angle here
    # because this axis is reversed in our definition of the FoV-system
    target_fov = SkyCoord(-lon, lat, frame=fov_frame)

    # Transform into celestial system (need not be ICRS)
    target_sky = target_fov.icrs

    return target_sky.ra, target_sky.dec


def sky_to_fov(lon, lat, lon_pnt, lat_pnt):
    """Transform sky coordinates to field-of-view coordinates.

    Parameters
    ----------
    lon, lat : `~astropy.units.Quantity`
        Sky coordinate to be transformed.
    lon_pnt, lat_pnt : `~astropy.units.Quantity`
        Coordinate specifying the pointing position.
        (i.e. the center of the field of view.)

    Returns
    -------
    lon_t, lat_t : `~astropy.units.Quantity`
        Transformed field-of-view coordinate.
    """
    # Create a frame that is centered on the pointing position
    center = SkyCoord(lon_pnt, lat_pnt)
    fov_frame = SkyOffsetFrame(origin=center)

    # Define coordinate to be transformed.
    target_sky = SkyCoord(lon, lat)

    # Transform into FoV-system
    target_fov = target_sky.transform_to(fov_frame)

    # Switch sign of longitude angle since this axis is
    # reversed in our definition of the FoV-system
    return -target_fov.lon, target_fov.lat
