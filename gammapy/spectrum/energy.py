# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.units import Quantity
from astropy.io import fits
from ..utils.array import array_stats_str

import astropy.units as u
import numpy as np

__all__ = ['Energy',
           'EnergyBounds']


class Energy(u.Quantity):

    """Energy bin centers.

    Stored as "ENERGIES" FITS table extensions.

    Parameters
    ----------
    energy : `~numpy.array`, scalar, `~astropy.units.Quantity`,
                        :class:`~gammapy.spectrum.energy.Energy`
        Energy

    unit : `~astropy.units.UnitBase`, str, optional
        The unit of the value specified for the energy.  This may be
        any string that `~astropy.units.Unit` understands, but it is
        better to give an actual unit object.

    dtype : `~numpy.dtype`, optional
        See `~astropy.units.Quantity`.

    copy : bool, optional
        See `~astropy.units.Quantity`.

    """

    def __new__(cls, energy, unit=None, dtype=None, copy=True):

        # Techniques to subclass Quantity taken from astropy.coordinates.Angle
        # see: http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        if unit is not None:
            unit = u.Unit(unit)
            if not unit.is_equivalent(u.eV):
                raise ValueError("Requested unit {0} is not an"
                                 " energy unit".format(unit))

        if isinstance(energy, u.Quantity):
            # also True if energy is of type Energy
            if unit is not None:
                energy = energy.to(unit).value
            else:
                unit = energy.unit
                if not unit.is_equivalent(u.eV):
                    raise ValueError("Given quantity {0} is not an"
                                     " energy".format(energy))
                energy = energy.value

        else:
            if unit is None:
                raise ValueError("No unit given")

        self = super(Energy, cls).__new__(cls, energy, unit,
                                          dtype=dtype, copy=copy)

        # Interesting once  energy bounds are stored
        self._nbins = self.size

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._nbins = self.size
        self._unit = getattr(obj, '_unit', None)

    def __quantity_subclass__(self, unit):
        unit = u.Unit(unit)
        if unit.is_equivalent(u.eV):
            return Energy, True
        else:
            return super(Energy, self).__quantity_subclass__(unit)[0], False

    @property
    def nbins(self):
        """
        The number of bins
        """
        return self._nbins

    @staticmethod
    def equal_log_spacing(emin, emax, nbins, unit=None):
        """Create Energy with equal log-spacing.

        if no unit is given, it will be taken from emax

        Parameters
        ----------
        emin : `~astropy.units.Quantity`, float
            Lowest energy bin
        emax : `~astropy.units.Quantity`, float
            Highest energy bin
        bins : int
            Number of bins
        unit : `~astropy.units.UnitBase`, str
            Energy unit

        Returns
        -------
        Energy
        """

        if unit is None:
            unit = emax.unit

        emin = Quantity(emin, unit)
        emax = Quantity(emax, unit)

        x_min, x_max = np.log10([emin.value, emax.value])
        energy = np.logspace(x_min, x_max, nbins)

        return Energy(energy, unit)

    @staticmethod
    def from_fits(hdu, unit=None):
        """Read ENERGIES fits extension.

        Parameters
        ---------
        hdu: `~astropy.io.fits.BinTableHDU`
            ``ENERGIES`` extensions.
        unit : `~astropy.units.UnitBase`, str
            Energy unit

        Returns
        ------
        Energy
        """

        header = hdu.header
        unit = header.get('TUNIT1')
        if unit is None:
            raise ValueError("No energy unit could be found in the header of."
                             "{0}. Please specifiy a unit".format(header.name))

        energy = Quantity(hdu.data['Energy'], unit)
        return Energy(energy)

    def to_fits(self, **kwargs):
        """Write ENERGIES fits extension

        Returns
        -------
        hdu: `~astropy.io.fits.BinTableHDU`
            ENERGIES fits extension


        """

        col1 = fits.Column(name='Energy', format='D', array=self.value)
        cols = fits.ColDefs([col1])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.name = 'ENERGIES'
        hdu.header['TUNIT1'] = "{0}".format(self.unit)

        return hdu


class EnergyBounds(Energy):

    """Energy bin edges

    Stored as "EBOUNDS" FITS table extensions.

    Parameters
    ----------

    energy : `~numpy.array`, scalar, `~astropy.units.Quantity`,
                        :class:`~gammapy.spectrum.energy.Energy`
        EnergyBounds
    unit : `~astropy.units.UnitBase`, str
        The unit of the values specified for the energy.  This may be any
        string that `~astropy.units.Unit` understands, but it is better to
        give an actual unit object.

    """

    def __new__(cls, energy, unit=None, dtype=None, copy=True):

        self = super(EnergyBounds, cls).__new__(cls, energy, unit,
                                                dtype=dtype, copy=copy)

        self._nbins = self.size - 1

        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._nbins = self.size - 1
        self._unit = getattr(obj, '_unit', None)

    @property
    def log_center(self):
        """Log centers of the energy bounds
        """
        
        center = np.sqrt(self[:-1] * self[1:])
        return Energy(center, self.unit)

    @staticmethod
    def equal_log_spacing(emin, emax, nbins, unit=None):
        """Create EnergyBounds with equal log-spacing.

        If no unit is given, it will be taken from emax

        Parameters
        ----------
        emin : `~astropy.units.Quantity`, float
            Lowest energy bin
        emax : `~astropy.units.Quantity`, float
            Highest energy bin
        bins : int
            Number of bins
        unit : `~astropy.units.UnitBase`, str
            Energy unit

        Returns
        -------
        EnergyBounds
        """

        bounds = super(EnergyBounds, EnergyBounds).equal_log_spacing(
            emin, emax, nbins + 1, unit)

        return bounds.view(EnergyBounds)

    @staticmethod
    def from_fits(hdu, unit=None):
        """Read EBOUNDS fits extension.

        Parameters
        ---------
        hdu: `~astropy.io.fits.BinTableHDU`
            ``EBOUNDS`` extensions.
        unit : `~astropy.units.UnitBase`, str
            Energy unit

        Returns
        ------
        Energy
        """

        header = hdu.header
        unit = header.get('TUNIT1')
        if unit is None:
            raise ValueError("No energy unit could be found in the header of."
                             "{0}. Please specifiy a unit".format(header.name))

        energy = Quantity(hdu.data['Energy'], unit)
        return Energy(energy)

    def to_fits(self, **kwargs):
        """Write EBOUNDS fits extension

        Returns
        -------
        hdu: `~astropy.io.fits.BinTableHDU`
            EBOUNDS fits extension


        """

        col1 = fits.Column(name='Energy', format='D', array=self.value)
        cols = fits.ColDefs([col1])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.name = 'EBOUNDS'
        hdu.header['TUNIT1'] = "{0}".format(self.unit)

        return hdu
