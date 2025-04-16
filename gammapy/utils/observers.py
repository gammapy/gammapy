# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Location of gamma-ray observatories."""

import astropy.units as u
from astropy.coordinates import EarthLocation
from gammapy.utils.deprecation import deprecated_key

__all__ = ["observatory_locations"]


class Observatories:
    """Gamma-ray observatory locations.

    Class that defines a dict with observatory names as keys
    and values of type `~astropy.coordinates.EarthLocation`.

    Note that with ``EarthLocation`` the orientation of angles is as follows:

    - longitude is east for positive values and west for negative values
    - latitude is north for positive values and south for negative values

    Available observatories (alphabetical order):

    - ``ctao_south`` and ``ctao_north`` for CTAO, see
      `Website <https://www.cta-observatory.org/>`__ and
      `Wikipedia <https://en.wikipedia.org/wiki/Cherenkov_Telescope_Array>`__
    - ``hawc`` for HAWC, see
      `Website <https://www.hawc-observatory.org/>`__ and
      `Wikipedia <https://en.wikipedia.org/wiki/High_Altitude_Water_Cherenkov_Experiment>`__
    - ``hegra`` for HEGRA, see `Wikipedia <https://en.wikipedia.org/wiki/HEGRA>`__
    - ``hess`` for HESS, see
      `Website <https://www.mpi-hd.mpg.de/hfm/HESS/>`__ and
      `Wikipedia <https://en.wikipedia.org/wiki/HESS>`__
    - ``magic`` for MAGIC, see
      `Website <https://wwwmagic.mpp.mpg.de/>`__ and
      `Wikipedia <https://en.wikipedia.org/wiki/MAGIC_(telescope)>`__
    - ``milagro`` for MILAGRO, see
      `Wikipedia <https://en.wikipedia.org/wiki/Milagro_(experiment)>`__)
    - ``veritas`` for VERITAS, see
      `Website <https://veritas.sao.arizona.edu/>`__ and
      `Wikipedia <https://en.wikipedia.org/wiki/VERITAS>`__
    - ``whipple`` for WHIPPLE, see
      `Wikipedia <https://en.wikipedia.org/wiki/Fred_Lawrence_Whipple_Observatory>`__

    Examples
    --------

    >>> from gammapy.data import observatory_locations
    >>> observatory_locations['hess']
    >>> list(observatory_locations.keys())
    """

    def __init__(self):
        self._data = {
            # Values from https://www.ctao.org/emission-to-discovery/array-sites/ctao-south/
            # Latitude: 24d41m0.34s South, Longitude: 70d18m58.84s West, Height: not given
            # Email from Gernot Maier on Sep 8, 2017, stating what they use in the CTA MC group:
            # lon=-70.31634499364885d, lat=-24.68342915473787d, height=2150m
            "ctao_south": EarthLocation(
                lon="-70d18m58.84s", lat="-24d41m0.34s", height="2150m"
            ),
            # Values from https://www.ctao.org/emission-to-discovery/array-sites/ctao-north/
            # Latitude: 28d45m43.7904s North, Longitude: 17d53m31.218s West, Height: 2200 m
            # Email from Gernot Maier on Sep 8, 2017, stating what they use in the CTA MC group for MST-1:
            # lon=-17.891571d, lat=28.762158d, height=2147m
            "ctao_north": EarthLocation(
                lon="-17d53m31.218s", lat="28d45m43.7904s", height="2147m"
            ),
            # HAWC location taken from https://arxiv.org/pdf/1108.6034v2.pdf
            "hawc": EarthLocation(lon="-97d18m34s", lat="18d59m48s", height="4100m"),
            # https://en.wikipedia.org/wiki/HEGRA
            "hegra": EarthLocation(lon="28d45m42s", lat="17d53m27s", height="2200m"),
            # Precision position of HESS from the HESS software (slightly different from Wikipedia)
            "hess": EarthLocation(
                lon="16d30m00.8s", lat="-23d16m18.4s", height="1835m"
            ),
            "magic": EarthLocation(lon="-17d53m24s", lat="28d45m43s", height="2200m"),
            "milagro": EarthLocation(
                lon="-106.67625d", lat="35.87835d", height="2530m"
            ),
            "veritas": EarthLocation(
                lon="-110d57m07.77s", lat="31d40m30.21s", height="1268m"
            ),
            # WHIPPLE coordinates taken from the Observatory Wikipedia page:
            # https://en.wikipedia.org/wiki/Fred_Lawrence_Whipple_Observatory
            "whipple": EarthLocation(
                lon="-110d52m42s", lat="31d40m52s", height="2606m"
            ),
            # communication with ASTRI Project Manager
            "astri": EarthLocation(
                lon="-16d30m20.99s", lat="28d18m00.0s", height="2370m"
            ),
            # coordinates from fact-tools (based on google earth)
            "fact": EarthLocation(
                lat=28.761647 * u.deg,
                lon=-17.891116 * u.deg,
                height=2200 * u.m,
            ),
        }

    def __getitem__(self, key):
        old_keys = ["cta_south", "cta_north"]
        new_keys = ["ctao_south", "ctao_north"]
        key = deprecated_key(key, old_keys, new_keys, since="2.0")
        return self._data[key]

    def __setitem__(self, key, value):
        # Handle deprecated keys when setting an item
        old_keys = ["cta_south", "cta_north"]
        new_keys = ["ctao_south", "ctao_north"]
        key = deprecated_key(key, old_keys, new_keys, since="2.0")
        self._data[key] = value

    def __delitem__(self, key):
        # Handle deprecated keys when deleting an item
        old_keys = ["cta_south", "cta_north"]
        new_keys = ["ctao_south", "ctao_north"]
        key = deprecated_key(key, old_keys, new_keys, since="2.0")
        del self._data[key]

    def __contains__(self, key):
        # Handle deprecated keys when checking existence
        old_keys = ["cta_south", "cta_north"]
        new_keys = ["ctao_south", "ctao_north"]
        key = deprecated_key(key, old_keys, new_keys, since="2.0")
        return key in self._data

    def keys(self):
        """Return the keys of the dictionary."""
        return list(self._data.keys())

    def values(self):
        """Return the values of the dictionary."""
        return list(self._data.values())

    def items(self):
        """Return the items (key-value pairs) of the dictionary."""
        return list(self._data.items())

    def get(self, key, default=None):
        """Return the value for the given key, or default if not found."""
        key = deprecated_key(
            key, ["cta_south", "cta_north"], ["ctao_south", "ctao_north"], since="2.0"
        )
        return self._data.get(key, default)

    def get_all(self):
        """Return the entire dictionary."""
        return self._data


observatory_locations = Observatories()
