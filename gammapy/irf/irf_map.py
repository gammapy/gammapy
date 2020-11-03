from copy import deepcopy
import numpy as np
from astropy.io import fits
from gammapy.maps import Map


class IRFMap:
    """IRF map base class"""

    def __init__(self, irf_map, exposure_map):
        self._irf_map = irf_map
        self.exposure_map = exposure_map

    @classmethod
    def from_hdulist(
        cls,
        hdulist,
        hdu=None,
        hdu_bands=None,
        exposure_hdu=None,
        exposure_hdu_bands=None,
    ):
        """Create from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdulist : `~astropy.fits.HDUList`
            HDU list.
        hdu : str
            Name or index of the HDU with the IRF map.
        hdu_bands : str
            Name or index of the HDU with the IRF map BANDS table.
        exposure_hdu : str
            Name or index of the HDU with the exposure map data.
        exposure_hdu_bands : str
            Name or index of the HDU with the exposure map BANDS table.

        Returns
        -------
        irf_map : `IRFMap`
            IRF map.
        """
        if hdu is None:
            hdu = cls._hdu_name

        irf_map = Map.from_hdulist(hdulist, hdu=hdu, hdu_bands=hdu_bands)

        if exposure_hdu is None:
            exposure_hdu = cls._hdu_name + "_exposure"

        if exposure_hdu in hdulist:
            exposure_map = Map.from_hdulist(
                hdulist, hdu=exposure_hdu, hdu_bands=exposure_hdu_bands
            )
        else:
            exposure_map = None

        return cls(irf_map, exposure_map)

    @classmethod
    def read(cls, filename, hdu=None):
        """Read an IRF_map from file and create corresponding object"""
        with fits.open(filename, memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    def to_hdulist(self):
        """Convert to `~astropy.io.fits.HDUList`.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list.
        """
        hdulist = self._irf_map.to_hdulist(hdu=self._hdu_name)

        exposure_hdu = self._hdu_name + "_exposure"

        if self.exposure_map is not None:
            new_hdulist = self.exposure_map.to_hdulist(hdu=exposure_hdu)
            hdulist.extend(new_hdulist[1:])

        return hdulist

    def write(self, filename, overwrite=False, **kwargs):
        """Write IRF map to fits"""
        hdulist = self.to_hdulist(**kwargs)
        hdulist.writeto(filename, overwrite=overwrite)

    def stack(self, other, weights=None):
        """Stack IRF map with another one in place.

        Parameters
        ----------
        other : `~gammapy.irf.IRFMap`
            IRF map to be stacked with this one.
        weights : `~gammapy.maps.Map`
            Map with stacking weights.

        """
        if self.exposure_map is None or other.exposure_map is None:
            raise ValueError(
                f"Missing exposure map for {self.__class__.__name__}.stack"
            )

        cutout_info = getattr(other._irf_map.geom, "cutout_info", None)

        if cutout_info is not None:
            slices = cutout_info["parent-slices"]
            parent_slices = Ellipsis, slices[0], slices[1]
        else:
            parent_slices = slice(None)

        self._irf_map.data[parent_slices] *= self.exposure_map.data[parent_slices]
        self._irf_map.stack(other._irf_map * other.exposure_map.data, weights=weights)

        # stack exposure map
        if weights and "energy" in weights.geom.axes.names:
            weights = weights.reduce(
                axis_name="energy", func=np.logical_or, keepdims=True
            )
        self.exposure_map.stack(other.exposure_map, weights=weights)

        with np.errstate(invalid="ignore"):
            self._irf_map.data[parent_slices] /= self.exposure_map.data[parent_slices]
            self._irf_map.data = np.nan_to_num(self._irf_map.data)

    def copy(self):
        """Copy IRF map"""
        return deepcopy(self)

    def cutout(self, position, width, mode="trim"):
        """Cutout IRF map.

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
        cutout : `IRFMap`
            Cutout IRF map.
        """
        irf_map = self._irf_map.cutout(position, width, mode)
        exposure_map = self.exposure_map.cutout(position, width, mode)
        return self.__class__(irf_map, exposure_map=exposure_map)

    def downsample(self, factor, axis=None, weights=None):
        """Downsample the spatial dimension by a given factor.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        axis : str
            Which axis to downsample. By default spatial axes are downsampled.
        weights : `~gammapy.maps.Map`
            Map with weights downsampling.

        Returns
        -------
        map : `IRFMap`
            Downsampled irf map.
        """
        irf_map = self._irf_map.downsample(
            factor=factor, axis=axis, preserve_counts=True, weights=weights
        )
        if axis is None:
            exposure_map = self.exposure_map.downsample(
                factor=factor, preserve_counts=False
            )
        else:
            exposure_map = self.exposure_map.copy()

        return self.__class__(irf_map, exposure_map=exposure_map)

    def slice_by_idx(self, slices):
        """Slice sub dataset.

        The slicing only applies to the maps that define the corresponding axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.

        Returns
        -------
        map_out : `IRFMap`
            Sliced irf map object.
        """
        irf_map = self._irf_map.slice_by_idx(slices=slices)
        if "energy_true" in slices:
            exposure_map = self.exposure_map.slice_by_idx(slices=slices)
        else:
            exposure_map = self.exposure_map
        return self.__class__(irf_map, exposure_map=exposure_map)
