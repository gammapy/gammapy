from pathlib import Path
from typing import Annotated
from astropy.coordinates import Angle, EarthLocation, SkyCoord
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


def json_encode_quantity(v):
    """JSON encoder for `~astropy.units.Quantity`."""
    return f"{v.value} {v.unit}"


def json_encode_angle(v):
    """JSON encoder for `~astropy.coordinates.Angle`."""
    return f"{v.value} {v.unit}"


def json_encode_time(v):
    """JSON encoder for `~astropy.time.Time`."""
    return f"{v.value}"


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


AngleType = Annotated[
    Angle,
    PlainSerializer(json_encode_angle, return_type=str, when_used="json-unless-none"),
    BeforeValidator(validate_angle),
]

EnergyType = Annotated[
    Quantity,
    PlainSerializer(
        json_encode_quantity, return_type=str, when_used="json-unless-none"
    ),
    BeforeValidator(validate_energy),
]

TimeType = Annotated[
    Time,
    PlainSerializer(json_encode_time, return_type=str, when_used="json-unless-none"),
    BeforeValidator(validate_time),
]


PathType = Annotated[Path, BeforeValidator(make_path)]


EarthLocationType = Annotated[
    EarthLocation,
    PlainSerializer(
        json_encode_earth_location, return_type=str, when_used="json-unless-none"
    ),
    BeforeValidator(validate_earth_location),
]

SkyCoordType = Annotated[
    SkyCoord,
    PlainSerializer(
        json_encode_sky_coord, return_type=str, when_used="json-unless-none"
    ),
    BeforeValidator(validate_sky_coord),
]
