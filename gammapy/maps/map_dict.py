# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import UserDict
from astropy.io import fits
from gammapy.maps import Map
from gammapy.utils.scripts import make_path

__all__ = ["MapDict"]

class MapDict(UserDict):
    def __init__(self, **kwargs):
        self._geom = None
        super().__init__(**kwargs)

    @property
    def geom(self):
        """Map geometry (`Geom`)"""
        return self._geom

    def __setitem__(self, key, value, **kwargs):
        if value is not None and not isinstance(value, Map):
            raise ValueError(f"MapDict can only contain Map objects.")

        if self.__len__() > 0:
            if value.geom != self._geom:
                raise ValueError(f"MapDict items must share the same geometry.")
        else:
            self._geom = value.geom

        super().__setitem__(key, value, **kwargs)
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        return f"{type(self).__name__}({self.data})"

    def to_hdulist(self, hdu_bands='BANDS'):
        """Convert map dictionary to list of HDUs.

        Parameters
        ----------
        hdu_bands : str
            Name of the HDU with the BANDS table. Default is 'BANDS'
            If set to None, each map will have its own hdu_band

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            Map dataset list of HDUs.
        """
        exclude_primary = slice(1, None)

        hdu_primary = fits.PrimaryHDU()
        hdulist = fits.HDUList([hdu_primary])

        for key, m in self.items():
            hdulist += m.to_hdulist(hdu=key, hdu_bands=hdu_bands)[
                exclude_primary
            ]

        return hdulist

    @classmethod
    def from_hdulist(cls, hdulist, hdu_bands='BANDS'):
        """Create map dictionary from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.
        hdu_bands : str
            Name of the HDU with the BANDS table. Default is 'BANDS'
            If set to None, each map should have its own hdu_band

        Returns
        -------
        map_Dict : `~gammapy.maps.MapDict`
            Map dict object.
        """
        maps = cls()

        for hdu in hdulist:
            if hdu.is_image and hdu.data is not None:
                map_name = hdu.name
                maps[map_name] = Map.from_hdulist(
                    hdulist, hdu=map_name, hdu_bands=hdu_bands
                )
        return maps

    @classmethod
    def read(cls, filename):
        """Read map dictionary from file.

        Parameters
        ----------
        filename : str
            Filename to read from.

        Returns
        -------
        flux_maps : `~gammapy.estimators.FluxMaps`
            Flux maps object.
        """
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist)

    def write(self, filename, overwrite=False):
        """Write map dictionary to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool
            Overwrite file if it exists.
        """
        filename = make_path(filename)

        hdulist = self.to_hdulist()
        hdulist.writeto(filename, overwrite=overwrite)

