import json
from pathlib import Path
from typing import Annotated
from astropy import units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time
from pydantic import PlainSerializer
from pydantic.functional_validators import AfterValidator, BeforeValidator
from .observers import observatory_locations
from .scripts import make_path

__all__ = [
    "AngleType",
    "EnergyType",
    "QuantityType",
    "TimeType",
    "PathType",
    "EarthLocationType",
    "SkyCoordType",
]


# TODO: replace by QuantityType and pydantic TypeAdapter
class JsonQuantityEncoder(json.JSONEncoder):
    """Support for quantities that JSON default encoder"""

    def default(self, obj):
        if isinstance(obj, u.Quantity):
            return obj.to_string()

        return json.JSONEncoder.default(self, obj)


# TODO: replace by QuantityType and pydantic TypeAdapter
class JsonQuantityDecoder(json.JSONDecoder):
    """Support for quantities that JSON default encoder"""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(data):
        for key, value in data.items():
            try:
                data[key] = u.Quantity(value)
            except TypeError:
                continue
        return data


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


def json_encode_time(v):
    """JSON encoder for `~astropy.time.Time`."""
    if v.isscalar:
        return v.value

    return v.value.tolist()


def validate_angle(v):
    """Validator for `~astropy.coordinates.Angle`."""
    return Angle(v)


def validate_scalar(v):
    """Validator for scalar values."""
    if not v.isscalar:
        raise ValueError(f"A scalar value is required: {v!r}")

    return v


def validate_energy(v):
    """Validator for `~astropy.units.Quantity` with unit "energy"."""
    v = u.Quantity(v)

    if v.unit.physical_type != "energy":
        raise ValueError(f"Invalid unit for energy: {v.unit!r}")

    return v


def validate_quantity(v):
    """Validator for `~astropy.units.Quantity`."""
    return u.Quantity(v)


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

scalar_validator = AfterValidator(validate_scalar)


AngleType = Annotated[
    Angle,
    PlainSerializer(lambda v: f"{v.value} {v.unit}", **SERIALIZE_KWARGS),
    BeforeValidator(validate_angle),
    scalar_validator,
]

EnergyType = Annotated[
    u.Quantity,
    PlainSerializer(lambda v: f"{v.value} {v.unit}", **SERIALIZE_KWARGS),
    BeforeValidator(validate_energy),
    scalar_validator,
]

QuantityType = Annotated[
    u.Quantity,
    PlainSerializer(lambda v: f"{v.value} {v.unit}", **SERIALIZE_KWARGS),
    BeforeValidator(validate_quantity),
    scalar_validator,
]

TimeType = Annotated[
    Time,
    PlainSerializer(json_encode_time, when_used="json-unless-none"),
    BeforeValidator(validate_time),
]


EarthLocationType = Annotated[
    EarthLocation,
    PlainSerializer(json_encode_earth_location, **SERIALIZE_KWARGS),
    BeforeValidator(validate_earth_location),
    scalar_validator,
]

SkyCoordType = Annotated[
    SkyCoord,
    PlainSerializer(json_encode_sky_coord, **SERIALIZE_KWARGS),
    BeforeValidator(validate_sky_coord),
    scalar_validator,
]

ICRSSkyCoordType = Annotated[
    SkyCoord,
    PlainSerializer(json_encode_sky_coord, **SERIALIZE_KWARGS),
    BeforeValidator(validate_sky_coord_icrs),
    scalar_validator,
]

AltAzSkyCoordType = Annotated[
    SkyCoord,
    PlainSerializer(json_encode_sky_coord, **SERIALIZE_KWARGS),
    BeforeValidator(validate_altaz_coord),
    scalar_validator,
]

PathType = Annotated[
    Path,
    PlainSerializer(lambda p: str(p), **SERIALIZE_KWARGS),
    BeforeValidator(make_path),
]
