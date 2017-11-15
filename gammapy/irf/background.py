# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from astropy.io import fits
import astropy.units as u
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.scripts import make_path
from ..utils.fits import fits_table_to_table
import numpy as np
__all__ = [
    'Background3D',
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
    >>> filename = '$GAMMAPY_EXTRA/test_datasets/cta_1dc/caldb/data/cta/prod3b/bcf/South_z20_50h/irf_file.fits'
    >>> bkg_3d = Background3D.read(filename, hdu='BACKGROUND')
    >>> print(bkg_3d)
    Background3D
    NDDataArray summary info
    energy         : size =    21, min =  0.016 TeV, max = 158.489 TeV
    detx           : size =    12, min = -5.500 deg, max =  5.500 deg
    dety           : size =    12, min = -5.500 deg, max =  5.500 deg
    Data           : size =  3024, min =  0.000 1 / (MeV s sr), max =  0.269 1 / (MeV s sr)
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

    @property
    def energy(self):
        return self.data.axis('energy')

    @property
    def detx(self):
        return self.data.axis('detx')

    @property
    def dety(self):
        return self.data.axis('dety')

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

    def to_table(self, provenance=None):
        """ Convert data to bintable HDU in CTA IRF format

        Data format specification: :ref:`gadf:edisp_2d`

        Parameters
        ----------
        provenance : `list`
            dict containing required information for fits header

        Examples
        --------
        Read energy dispersion IRF from disk:
        from gammapy.irf import Background3D
        
        head = ([  
            ('ORIGIN', 'IRAP', 'Name of organization making this file'),
            ('DATE', '2017-09-27T12:02:24', 'File creation date (YYYY-MM-DDThh:mm:ss UTC)'),
            ('TELESCOP', 'CTA', 'Name of telescope'),
            ('INSTRUME', 'PROD3B', 'Name of instrument'),
            ('DETNAM', 'NONE', 'Name of detector'),
            ('HDUCLASS', 'OGIP', 'HDU class'),
            ('HDUDOC', '???', 'HDU documentation'),
            ('HDUCLAS1', 'RESPONSE', 'HDU class'),
            ('HDUCLAS2', 'BKG', 'HDU class)
            ...])
            
        bkg = Background3D(detx_lo, detx_hi, dety_lo, dety_hi, e_true_lo, e_true_hi, data)
        hdu = bgk.to_table(head)
        prim_hdu = fits.PrimaryHDU()
        fits.HDUList([prim_hdu, hdu]).writeto('irffile.fits')
        """
        print('UNDERGOING CONSTRUCTION')
        c1 = fits.Column(name='DETX_LO', array=np.asarray([self.detx.lo]),
                         format='{}E'.format(self.detx.nbins), unit='{}'.format(self.detx.unit))
        c2 = fits.Column(name='DETX_HI', array=np.asarray([self.detx.hi]),
                         format='{}E'.format(self.detx.nbins), unit='{}'.format(self.detx.unit))
        c3 = fits.Column(name='DETY_LO', array=np.asarray([self.dety.lo]),
                         format='{}E'.format(self.dety.nbins), unit='{}'.format(self.detx.unit))
        c4 = fits.Column(name='DETY_HI', array=np.asarray([self.dety.hi]),
                         format='{}E'.format(self.dety.nbins), unit='{}'.format(self.detx.unit))
        c5 = fits.Column(name='ENERGY_LO', array=np.asarray([self.energy.lo]),
                         format='{}E'.format(self.energy.nbins), unit='{}'.format(self.energy.unit))
        c6 = fits.Column(name='ENERGY_HI', array=np.asarray([self.energy.hi]),
                         format='{}E'.format(self.energy.nbins), unit='{}'.format(self.energy.unit))
        c7 = fits.Column(name='BGD', array=np.asarray([self.data.data]),
                         format='{}E'.format(self.energy.nbins * self.detx.nbins * self.dety.nbins),
                         dim='({},{},{})'.format(self.detx.nbins, self.dety.nbins, self.energy.nbins),
                         unit='{}'.format(self.data.data.unit))
        # self.provenance()
        header = fits.Header()
        header.update(provenance)
        table = fits.BinTableHDU.from_columns([c1, c2, c3, c4, c5, c6, c7], header=header, name='BACKGROUND')

        return table
