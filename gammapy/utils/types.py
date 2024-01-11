from pathlib import Path
from typing import Annotated
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity
from pydantic import PlainSerializer
from pydantic.functional_validators import BeforeValidator
from .observers import observatory_locations
from .scripts import make_path

__all__ = [
    "AngleType",
    "EnergyType",
    "TimeType",
    "PathType",
    "EarthLocationType",
    "SkyCoordType",
]


def json_encode_earth_location(v):
    """JSON encoder for `~astropy.coordinates.EarthLocation`."""
    return (
        f"lon: {v.lon.value} {v.lon.unit}, "
        f"lat : {v.lat.value} {v.lat.unit}, "
        f"height : {v.height.value} {v.height.unit}"
    )


def json_encode_sky_coord(v):
    """JSON encoder for `~astropy.coordinates.SkyCoord`."""
    return f"lon: {v.spherical.lon.value} {v.spherical.lon.unit}, lat: {v.spherical.lat.value} {v.spherical.lat.unit}, frame: {v.frame.name} "


def validate_angle(v):
    """Validator for `~astropy.coordinates.Angle`."""
    return Angle(v)


def validate_energy(v):
    """Validator for `~astropy.units.Quantity` with unit "energy"."""
    v = Quantity(v)

    if v.unit.physical_type != "energy":
        raise ValueError(f"Invalid unit for energy: {v.unit!r}")

    return v


def validate_time(v):
    """Validator for `~astropy.time.Time`."""
    return Time(v)


def validate_earth_location(v):
    """Validator for `~astropy.coordinates.EarthLocation`."""
    if isinstance(v, EarthLocation):
        return v

    if isinstance(v, str) and v in observatory_locations:
        return observatory_locations[v]

    try:
        return EarthLocation(v)
    except TypeError:
        raise ValueError(f"Invalid EarthLocation: {v!r}")


def validate_sky_coord(v):
    """Validator for `~astropy.coordinates.SkyCoord`."""
    return SkyCoord(v)


def validate_sky_coord_icrs(v):
    """Validator for `~astropy.coordinates.SkyCoord` in icrs."""
    try:
        return SkyCoord(v).icrs
    except AttributeError:
        raise ValueError(f"Cannot convert '{v!r}' to icrs")


def validate_altaz_coord(v):
    """Validator for `~astropy.coordinates.AltAz`."""
    if isinstance(v, AltAz):
        return SkyCoord(v)

    return SkyCoord(v).altaz


SERIALIZE_KWARGS = {
    "when_used": "json-unless-none",
    "return_type": str,
}


AngleType = Annotated[
    Angle,
    PlainSerializer(lambda v: f"{v.value} {v.unit}", **SERIALIZE_KWARGS),
    BeforeValidator(validate_angle),
]

EnergyType = Annotated[
    Quantity,
    PlainSerializer(lambda v: f"{v.value} {v.unit}", **SERIALIZE_KWARGS),
    BeforeValidator(validate_energy),
]

TimeType = Annotated[
    Time,
    PlainSerializer(lambda v: f"{v.value}", **SERIALIZE_KWARGS),
    BeforeValidator(validate_time),
]


EarthLocationType = Annotated[
    EarthLocation,
    PlainSerializer(json_encode_earth_location, **SERIALIZE_KWARGS),
    BeforeValidator(validate_earth_location),
]

SkyCoordType = Annotated[
    SkyCoord,
    PlainSerializer(json_encode_sky_coord, **SERIALIZE_KWARGS),
    BeforeValidator(validate_sky_coord),
]

ICRSSkyCoordType = Annotated[
    SkyCoord,
    PlainSerializer(json_encode_sky_coord, **SERIALIZE_KWARGS),
    BeforeValidator(validate_sky_coord_icrs),
]

AltAzSkyCoordType = Annotated[
    SkyCoord,
    PlainSerializer(json_encode_sky_coord, **SERIALIZE_KWARGS),
    BeforeValidator(validate_altaz_coord),
]

PathType = Annotated[
    Path,
    PlainSerializer(lambda p: str(p), **SERIALIZE_KWARGS),
    BeforeValidator(make_path),
]
