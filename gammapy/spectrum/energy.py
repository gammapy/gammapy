# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.units import Quantity
from ..utils.array import array_stats_str

import astropy.units as u


__all__= ['Energy']



class Energy(Quantity):
    """Energy bin centers.
    
    Stored as "ENERGIES" FITS table extensions.

    Parameters
    ----------

    energy : `~numpy.ndarray`
        Energy
    unit : `~astropy.units.UnitBase`, str
        The unit of the values specified for the energy.  This may be any
        string that `~astropy.units.Unit` understands, but it is better to
        give an actual unit object.

    """

    def __init__(self, energy, unit=None):

        if isinstance(energy, Quantity):
            energy = energy.value
            unit = energy.unit

        if isinstance(unit, u.UnitBase):
            pass
        elif isinstance(unit, basestring):
            unit = u.Unit(unit)
        elif unit is None:
            raise UnitsError("No unit was specified in Energy initializer")

        
        self._unit = unit
        self._value = energy


    @staticmethod
    def equal_log_spacing(emin, emax, nbins):
        """Create Energy with equal log-spacing.

        The unit will be taken from emax

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Lowest energy bin
        emax : `~astropy.units.Quantity`
            Highest energy bin
        bins : int
            Number of bins

        Returns
        -------
        Energy
        """

        if not isinstance(emin, Quantity) or not isinstance(emax, Quantity):
            raise ValueError("Energies must be Quantities")

        x_min, x_max = np.log10([emin.value, emax.value])
        energy = np.logspace(x_min, x_max, nbins + 1)
        energy = Quantity(energy, emax.unit)

        return Energy(energy)

    @staticmethod
    def from_fits(hdulist):
        """Read ENERGIES fits extension.

        Parameters
        ---------
        hdu_list : `~astropy.io.fits.HDUList` 
            HDU list with ``SPECRESP`` extensions.

        Returns
        ------
        Energy
        """

        energy = Quantity(hdulist['ENERGIES'].data['Energy'], 'MeV')
        return Energy(energy)


    def to_fits(self, **kwargs):
        """Write ENERGIES fits extension

        Returns
        -------
        hdu_list : `~astropy.io.fits.BinTableHDU`
            ENERGIES fits extension


        """

        col1 = fits.Column(name='Energy', format='D', array=self.value)
        cols = fits.ColDefs([col1])
        return fits.BinTableHDU.from_columns(cols)


    def info(self):
        #is this really necessary? what's the benefit towards just typing the
        #name of the instance?
        pass
