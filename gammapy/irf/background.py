# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import Angle
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.scripts import make_path
from ..utils.fits import fits_table_to_table, table_to_fits_table
from ..utils.energy import EnergyBounds
import numpy as np

__all__ = [
    'Background3D',
    'Background2D'
]


class Background3D(object):
    """Background 3D.

    Data format specification: :ref:`gadf:bkg_3d`

    Parameters
    -----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Energy binning
    detx_lo, detx_hi : `~astropy.units.Quantity`
        FOV coordinate X-axis binning
    dety_lo, dety_hi : `~astropy.units.Quantity`
        FOV coordinate Y-axis binning
    data : `~astropy.units.Quantity`
        Background rate (usually: ``s^-1 MeV^-1 sr^-1``)

    Examples
    --------
    Here's an example you can use to learn about this class:

    >>> from gammapy.irf import Background3D
    >>> filename = '$GAMMAPY_EXTRA/datasets/cta-1dc/caldb/data/cta//1dc/bcf/South_z20_50h/irf_file.fits'
    >>> bkg_3d = Background3D.read(filename, hdu='BACKGROUND')
    >>> print(bkg_3d)
    Background3D
    NDDataArray summary info
    energy         : size =    21, min =  0.016 TeV, max = 158.489 TeV
    detx           : size =    36, min = -5.833 deg, max =  5.833 deg
    dety           : size =    36, min = -5.833 deg, max =  5.833 deg
    Data           : size = 27216, min =  0.000 1 / (MeV s sr), max =  0.421 1 / (MeV s sr)
    """
    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
    """Default Interpolation kwargs for `~NDDataArray`. Extrapolate."""

    def __init__(self, energy_lo, energy_hi,
                 detx_lo, detx_hi, dety_lo, dety_hi,
                 data, meta=None, interp_kwargs=None):

        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi,
                interpolation_mode='log', name='energy'),
            BinnedDataAxis(
                detx_lo, detx_hi,
                interpolation_mode='linear', name='detx'),
            BinnedDataAxis(
                dety_lo, dety_hi,
                interpolation_mode='linear', name='dety'),
        ]
        self.data = NDDataArray(axes=axes, data=data,
                                interp_kwargs=interp_kwargs)
        self.meta = OrderedDict(meta) if meta else OrderedDict()

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\n{}'.format(self.data)
        return ss

    @classmethod
    def from_table(cls, table):
        """Read from `~astropy.table.Table`."""
        # Spec says key should be "BKG", but there are files around
        # (e.g. CTA 1DC) that use "BGD". For now we support both
        if 'BKG' in table.colnames:
            bkg_name = 'BKG'
        elif 'BGD' in table.colnames:
            bkg_name = 'BGD'
        else:
            raise ValueError('Invalid column names. Need "BKG" or "BGD".')

        # Currently some files (e.g. CTA 1DC) contain unit in the FITS file
        # '1/s/MeV/sr', which is invalid ( try: astropy.unit.Unit('1/s/MeV/sr')
        # This should be corrected.
        # For now, we hard-code the unit here:
        data_unit = u.Unit('s-1 MeV-1 sr-1')

        return cls(
            energy_lo=table['ENERG_LO'].quantity[0],
            energy_hi=table['ENERG_HI'].quantity[0],
            detx_lo=table['DETX_LO'].quantity[0],
            detx_hi=table['DETX_HI'].quantity[0],
            dety_lo=table['DETY_LO'].quantity[0],
            dety_hi=table['DETY_HI'].quantity[0],
            data=table[bkg_name].data[0] * data_unit,
            meta=table.meta,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu='BACKGROUND'):
        """Create from `~astropy.io.fits.HDUList`."""
        fits_table = hdulist[hdu]
        table = fits_table_to_table(fits_table)
        return cls.from_table(table)

    @classmethod
    def read(cls, filename, hdu='BACKGROUND'):
        """Read from file."""
        filename = make_path(filename)
        hdulist = fits.open(str(filename))
        return cls.from_hdulist(hdulist, hdu=hdu)

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        meta = self.meta.copy()
        table = Table(meta=meta)
        table['DETX_LO'] = self.data.axis('detx').lo[np.newaxis]
        table['DETX_HI'] = self.data.axis('detx').hi[np.newaxis]
        table['DETY_LO'] = self.data.axis('dety').lo[np.newaxis]
        table['DETY_HI'] = self.data.axis('dety').hi[np.newaxis]
        table['ENERG_LO'] = self.data.axis('energy').lo[np.newaxis]
        table['ENERG_HI'] = self.data.axis('energy').hi[np.newaxis]
        table['BKG'] = self.data.data[np.newaxis]
        return table

    def to_hdulist(self, name='BACKGROUND'):
        """Convert to `~astropy.io.fits.BinTable`."""
        hdu = table_to_fits_table(self.to_table(), name)
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
        return hdulist

    def write(self, filename, name='BACKGROUND', **kwargs):
        """Convert to `~astropy.io.fits.BinTable`.
        Parameters
        ----------
        filename : str
            File name
        """
        hdulist = self.to_hdulist(name=name)
        hdulist.writeto(filename, **kwargs)

    def evaluate(self, fov_offset, fov_phi, energy_reco, method=None, **kwargs):
        """
        Evaluate the `Background3D` at a given FOV coordinate and energy.

        Parameters
        ----------
        fov_offset : `~astropy.coordinates.Angle`
            offset in the FOV. Same shape than fov_phi`
        fov_phi: `~astropy.coordinates.Angle`
            azimuth angle in the FOV. Same shape than `fov_offset`
        energy_reco : `~astropy.units.Quantity`
            Vector of energy (1D) on which the model is evaluated
        method : str {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values, axis order is the same as for the NDData array
        """

        detx = fov_offset * np.cos(fov_phi)
        dety = fov_offset * np.sin(fov_phi)

        if energy_reco is None:
            energy_reco = self.data.axis('energy').nodes

        array = self.data.evaluate(detx=detx, dety=dety, energy=energy_reco, method=method, **kwargs)
        return array


class Background2D(object):
    """Background 3D.

    Data format specification: :ref:`gadf:bkg_3d`

    Parameters
    -----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Energy binning
    offset_lo, offset_hi : `~astropy.units.Quantity`
        FOV coordinate Y-axis binning
    data : `~astropy.units.Quantity`
        Background rate (usually: ``s^-1 MeV^-1 sr^-1``)

    """
    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
    """Default Interpolation kwargs for `~NDDataArray`. Extrapolate."""

    def __init__(self, energy_lo, energy_hi,
                 offset_lo, offset_hi,
                 data, meta=None, interp_kwargs=None):

        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi,
                interpolation_mode='log', name='energy'),
            BinnedDataAxis(
                offset_lo, offset_hi,
                interpolation_mode='linear', name='offset'),
        ]
        self.data = NDDataArray(axes=axes, data=data,
                                interp_kwargs=interp_kwargs)
        self.meta = OrderedDict(meta) if meta else OrderedDict()

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\n{}'.format(self.data)
        return ss

    @classmethod
    def from_table(cls, table):
        """Read from `~astropy.table.Table`."""
        # Spec says key should be "BKG", but there are files around
        # (e.g. CTA 1DC) that use "BGD". For now we support both
        if 'BKG' in table.colnames:
            bkg_name = 'BKG'
        elif 'BGD' in table.colnames:
            bkg_name = 'BGD'
        else:
            raise ValueError('Invalid column names. Need "BKG" or "BGD".')

        # Currently some files (e.g. CTA 1DC) contain unit in the FITS file
        # '1/s/MeV/sr', which is invalid ( try: astropy.unit.Unit('1/s/MeV/sr')
        # This should be corrected.
        # For now, we hard-code the unit here:
        data_unit = u.Unit('s-1 MeV-1 sr-1')
        return cls(
            energy_lo=table['ENERG_LO'].quantity[0],
            energy_hi=table['ENERG_HI'].quantity[0],
            offset_lo=table['THETA_LO'].quantity[0],
            offset_hi=table['THETA_HI'].quantity[0],
            data=table[bkg_name].data[0] * data_unit,
            meta=table.meta,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu='BACKGROUND'):
        """Create from `~astropy.io.fits.HDUList`."""
        fits_table = hdulist[hdu]
        table = fits_table_to_table(fits_table)
        return cls.from_table(table)

    @classmethod
    def read(cls, filename, hdu='BACKGROUND'):
        """Read from file."""
        filename = make_path(filename)
        hdulist = fits.open(str(filename))
        return cls.from_hdulist(hdulist, hdu=hdu)

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        meta = self.meta.copy()
        table = Table(meta=meta)

        table['THETA_LO'] = self.data.axis('offset').lo[np.newaxis]
        table['THETA_HI'] = self.data.axis('offset').hi[np.newaxis]
        table['ENERG_LO'] = self.data.axis('energy').lo[np.newaxis]
        table['ENERG_HI'] = self.data.axis('energy').hi[np.newaxis]
        table['BKG'] = self.data.data[np.newaxis]
        return table

    def to_hdulist(self, name='BACKGROUND'):
        """Convert to `~astropy.io.fits.BinTable`."""
        hdu = table_to_fits_table(self.to_table(), name)
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
        return hdulist

    def write(self, filename, name='BACKGROUND', **kwargs):
        """Convert to `~astropy.io.fits.BinTable`.
        Parameters
        ----------
        filename : str
            File name
        """
        hdulist = self.to_hdulist(name=name)
        hdulist.writeto(filename, **kwargs)

    def evaluate(self, fov_offset, fov_phi=None, energy_reco=None, method=None, **kwargs):
        """
        Evaluate the `Background2D` at a given offset and energy.

        Parameters
        ----------
        fov_offset : `~astropy.coordinates.Angle`
            offset in the FOV
        fov_phi: `~astropy.coordinates.Angle`
            azimuth angle in the FOV. Not used for this class since the background model is radially symmetric
        energy_reco : `~astropy.units.Quantity`
            Vector of energy (1D) on which the model is evaluated
        method : str {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        array : `~astropy.units.Quantity`
            Interpolated values, axis order is the same as for the NDData array

        """
        if energy_reco is None:
            energy_reco = self.data.axis('energy').nodes

        array = self.data.evaluate(offset=fov_offset, energy=energy_reco, method=method, **kwargs)
        return array

    def integrate_on_energy_range(self, tab_energy_band, energy_bins, fov_offset, fov_phi=None, method=None, **kwargs):
        """
        Evaluate the `Background2D` at a given offset and integrate over energy_bands. In each energy_band, evaluate
        at different energies and then integrate in order to get the total background rate per steradian in the given
        energy range

        Parameters
        ----------
        tab_energy_band : `~astropy.units.Quantity`
            Energy bin edges (vector 1D)
        energy_bins : int
            Number of energy bins used for the integration
        fov_offset : `~astropy.coordinates.Angle`
            offset in the FOV
        fov_phi: `~astropy.coordinates.Angle`
            azimuth angle in the FOV. Not used for this class since the background model is radially symmetric
        method : str {'linear', 'nearest'}, optional
            Interpolation method
        kwargs : dict
            option for interpolation for `~scipy.interpolate.RegularGridInterpolator`

        Returns
        -------
        array : `~astropy.units.Quantity`
            Background rate per steradian, axis order is the same as for the NDData array
        """
        fov_offset = Angle(np.array(fov_offset), fov_offset.unit)
        n_energy_band = len(tab_energy_band) - 1
        shape_bkg = tuple([n_energy_band]) + fov_offset.shape
        bkg_integrated = u.Quantity(np.zeros(shape_bkg), "1 / (s sr)")
        for i_e, (e_min, e_max) in enumerate(zip(tab_energy_band[0:-1], tab_energy_band[1:])):
            energy_edges = EnergyBounds.equal_log_spacing(e_min, e_max, energy_bins)
            energy_bins = energy_edges.log_centers
            bkg_evaluated = self.evaluate(fov_offset=fov_offset, fov_phi=fov_phi, energy_reco=energy_bins,
                                          method=method, **kwargs)
            if len(fov_offset.shape) == 1:
                bkg_integrated[i_e, :] = np.sum(bkg_evaluated.T * energy_edges.bands, axis=1).T
            else:
                bkg_integrated[i_e, :, :] = np.sum(bkg_evaluated.T * energy_edges.bands, axis=2).T

        return bkg_integrated.squeeze()
