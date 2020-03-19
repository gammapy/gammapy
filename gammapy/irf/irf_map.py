from astropy.io import fits
from gammapy.maps import Map
import numpy as np
from copy import deepcopy


class IRFMap:
    """IRF map base class"""
    def __init__(self, irf_map, exposure_map):
        self._irf_map = irf_map
        self.exposure_map = exposure_map

    @classmethod
    def from_hdulist(cls, hdulist):
        """Convert to `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdulist : `~astropy.fits.HDUList`
            HDU list.
        """
        irf_map = Map.from_hdulist(hdulist, cls._hdu_name)

        exposure_hdu = cls._hdu_name + "_exposure"

        if exposure_hdu in hdulist:
            exposure_map = Map.from_hdulist(hdulist, exposure_hdu)
        else:
            exposure_map = None

        return cls(irf_map, exposure_map)

    @classmethod
    def read(cls, filename):
        """Read an edisp_map from file and create an EDispMap object"""
        with fits.open(filename, memmap=False) as hdulist:
            return cls.from_hdulist(hdulist)

    def to_hdulist(self):
        """Convert to `~astropy.io.fits.HDUList`.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
        """
        hdulist = self._irf_map.to_hdulist(hdu=self._hdu_name)

        exposure_hdu = self._hdu_name + "_exposure"

        if self.exposure_map is not None:
            new_hdulist = self.exposure_map.to_hdulist(hdu=exposure_hdu)
            hdulist.extend(new_hdulist[1:])

        return hdulist

    def write(self, filename, overwrite=False, **kwargs):
        """Write to fits"""
        hdulist = self.to_hdulist(**kwargs)
        hdulist.writeto(filename, overwrite=overwrite)

    def stack(self, other, weights=None):
        """Stack EDispMap with another one in place.

        Parameters
        ----------
        other : `~gammapy.cube.EDispMap`
            Energy dispersion map to be stacked with this one.
        weights : `~gammapy.maps.Map`
            Map with stacking weights.

        """
        if self.exposure_map is None or other.exposure_map is None:
            raise ValueError("Missing exposure map for PSFMap.stack")

        cutout_info = other._irf_map.geom.cutout_info

        if cutout_info is not None:
            slices = cutout_info["parent-slices"]
            parent_slices = Ellipsis, slices[0], slices[1]
        else:
            parent_slices = None

        self._irf_map.data[parent_slices] *= self.exposure_map.data[parent_slices]
        self._irf_map.stack(other._irf_map * other.exposure_map.data, weights=weights)

        # stack exposure map
        if weights and "energy" in weights.geom.axes_names:
            weights = weights.slice_by_idx({"energy": slice(0, 1)})

        self.exposure_map.stack(other.exposure_map, weights=weights)

        with np.errstate(invalid="ignore"):
            self._irf_map.data[parent_slices] /= self.exposure_map.data[parent_slices]
            self._irf_map.data = np.nan_to_num(self._irf_map.data)

    def copy(self):
        """Copy EDispMap"""
        return deepcopy(self)

    def cutout(self, position, width, mode="trim"):
        """Cutout edisp map.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : tuple of `~astropy.coordinates.Angle`
            Angular sizes of the region in (lon, lat) in that specific order.
            If only one value is passed, a square region is extracted.
        mode : {'trim', 'partial', 'strict'}
            Mode option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.

        Returns
        -------
        cutout : `EdispMap`
            Cutout edisp map.
        """
        irf_map = self._irf_map.cutout(position, width, mode)
        exposure_map = self.exposure_map.cutout(position, width, mode)
        return self.__class__(irf_map, exposure_map=exposure_map)
