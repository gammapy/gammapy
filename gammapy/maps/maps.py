# Licensed under a 3-clause BSD style license - see LICENSE.rst
import html
from collections.abc import MutableMapping
from astropy.io import fits
from gammapy.maps import Map
from gammapy.utils.scripts import make_path

__all__ = ["Maps"]


class Maps(MutableMapping):
    """A Dictionary containing Map objects sharing the same geometry.

    This class simplifies handling and I/O of maps collections.

    For maps with different geometries, use a regular dict.
    """

    def __init__(self, **kwargs):
        self._geom = None
        self._data = {}
        for key, value in kwargs.items():
            self.__setitem__(key, value)

    @property
    def geom(self):
        """Map geometry as a `~gammapy.maps.Geom` object."""
        return self._geom

    def __setitem__(self, key, value):
        if value is not None and not isinstance(value, Map):
            raise ValueError(
                f"MapDict can only contain Map objects, got {type(value)} instead."
            )
        # TODO: which loosers criterion to apply? broadcastability?
        else:
            self._geom = value.geom

        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        """Returns the length of MapDict."""
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"{type(self).__name__}({self._data})"

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"
        str_ += self._geom.__repr__()
        str_ += "\n"
        for name, value in self.items():
            str_ += f"{name} \n"
            str_ += f"\t unit\t : {value.unit} \n"
            str_ += f"\t dtype\t : {value.data.dtype}\n"
            str_ += "\n"
        return str_

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def to_hdulist(self, hdu_bands="BANDS"):
        """Convert map dictionary to list of HDUs.

        Parameters
        ----------
        hdu_bands : str, optional
            Name of the HDU with the BANDS table. If set to None, each map will have its own hdu_band.
            Default is 'BANDS'.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            Map dataset list of HDUs.
        """
        exclude_primary = slice(1, None)

        hdu_primary = fits.PrimaryHDU()
        hdulist = fits.HDUList([hdu_primary])

        for key, m in self.items():
            hdulist += m.to_hdulist(hdu=key, hdu_bands=hdu_bands)[exclude_primary]

        return hdulist

    @classmethod
    def from_hdulist(cls, hdulist, hdu_bands="BANDS"):
        """Create map dictionary from a HDU list.

        Because FITS keywords are case-insensitive, all key names will return as lower-case.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.
        hdu_bands : str, optional
            Name of the HDU with the BANDS table. If set to None, each map should have its own hdu_band.
            Default is 'BANDS'.

        Returns
        -------
        maps : `~gammapy.maps.Maps`
            Maps object.
        """
        maps = cls()

        for hdu in hdulist:
            if hdu.is_image and hdu.data is not None:
                map_name = hdu.name.lower()
                maps[map_name] = Map.from_hdulist(
                    hdulist, hdu=map_name, hdu_bands=hdu_bands
                )
        return maps

    @classmethod
    def read(cls, filename, checksum=False):
        """Read map dictionary from file.

        Because FITS keywords are case-insensitive, all key names will return as lower-case.

        Parameters
        ----------
        filename : str
            Filename to read from.

        Returns
        -------
        maps : `~gammapy.maps.Maps`
            Maps object.
        """
        with fits.open(
            str(make_path(filename)), memmap=False, checksum=checksum
        ) as hdulist:
            return cls.from_hdulist(hdulist)

    def write(self, filename, overwrite=False, checksum=False):
        """Write map dictionary to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool, optional
            Overwrite existing file. Default is False.
        checksum : bool, optional
            When True adds both DATASUM and CHECKSUM cards to the headers written to the file.
            Default is False.
        """
        filename = make_path(filename)

        hdulist = self.to_hdulist()
        hdulist.writeto(filename, overwrite=overwrite, checksum=checksum)

    @classmethod
    def from_geom(cls, geom, names, kwargs_list=None):
        """Create map dictionary from a geometry.

        Parameters
        ----------
        geom : `~gammapy.maps.Geom`
            The input geometry that will be used by all maps.
        names : list of str
            The list of all map names.
        kwargs_list : list of dict
            the list of arguments to be passed to `~gammapy.maps.Map.from_geom()`.

        Returns
        -------
        maps : `~gammapy.maps.Maps`
            Maps object.
        """
        mapdict = {}

        if kwargs_list is None:
            kwargs_list = [{}] * len(names)

        for name, kwargs in zip(names, kwargs_list):
            mapdict[name] = Map.from_geom(geom, **kwargs)

        return cls(**mapdict)
