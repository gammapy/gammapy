# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ..utils.scripts import make_path
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..irf import EffectiveAreaTable2D, EffectiveAreaTable
from ..background import FOVCube
from ..irf import EnergyDispersion2D, EnergyDispersion
from ..irf import EnergyDependentMultiGaussPSF


__all__ = [
    'CTAIrf',
    'BgRateTable',
    'Psf68Table',
    'CTAPerf'
]


class CTAIrf(object):
    """CTA instrument response function container.

    Class handling CTA instrument response function.

    For now we use the production 2 of the CTA IRF
    (https://portal.cta-observatory.org/Pages/CTA-Performance.aspx)
    adapted from the ctools
    (http://cta.irap.omp.eu/ctools/user_manual/getting_started/response.html).

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.
    Waiting for a new public production of the CTA IRF,
    we'll fix the missing pieces.

    This class is similar to `~gammapy.data.DataStoreObservation`,
    but only contains IRFs (no event data or livetime info).
    TODO: maybe re-factor code somehow to avoid code duplication.

    Parameters
    ----------
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion2D`
        Energy dispersion
    psf : `~gammapy.irf.EnergyDependentMultiGaussPSF`
        Point spread function
    bkg : `~gammapy.background.FOVCube`
        Background rate
    """

    def __init__(self, aeff=None, edisp=None, psf=None, bkg=None):
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg

    @classmethod
    def read(cls, filename):
        """
        Read from a FITS file.

        Parameters
        ----------
        filename : `str`
            File containing the IRFs
        """
        filename = str(make_path(filename))

        # table = Table.read(filename, hdu='EFFECTIVE AREA')
        # aeff = EffectiveAreaTable2D.from_table(table)
        aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')

        # TODO: fix `FOVCube.read`, then use it directly here.
        table = fits.open(filename)['BACKGROUND']
        table.columns.change_name(str('BGD'), str('Bgd'))
        table.header['TUNIT7'] = '1 / (MeV s sr)'
        bkg = FOVCube.from_fits_table(table, scheme='bg_cube')

        # Dealing energy dispersion matrix
        # Fix offset values a la gammapy (theta_lo=theta_hi)
        edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')

        # Dealing with psf and fix values to get a single gaussian component
        # by setting the amplitudes and the sigmas of the last two components
        # to 0 and 1, respectively
        # Fix offset values a la gammapy (theta_lo=theta_hi)
        table_hdu_psf = fits.open(filename)['POINT SPREAD FUNCTION']

        table_hdu_psf.data[0]['AMPL_2'] = 0
        table_hdu_psf.data[0]['AMPL_3'] = 0

        table_hdu_psf.data[0]['SIGMA_2'] = 1
        table_hdu_psf.data[0]['SIGMA_3'] = 1

        psf = EnergyDependentMultiGaussPSF.from_fits(table_hdu_psf)

        return cls(
            aeff=aeff,
            bkg=bkg,
            edisp=edisp,
            psf=psf,
        )


class BgRateTable(NDDataArray):
    """Background rate Table

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.
    Work will be done to fix this.

    Parameters
    -----------
    energy : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
        Bin edges of energy axis
    data : `~astropy.units.Quantity`
        Background rate
    """
    energy = BinnedDataAxis(interpolation_mode='log')
    """Energy Axis"""
    axis_names = ['energy']

    @classmethod
    def from_table(cls, table):
        """Background rate reader"""
        energy_col = 'ENERG'
        data_col = 'BGD'

        energy_lo = table['{}_LO'.format(energy_col)].quantity
        energy_hi = table['{}_HI'.format(energy_col)].quantity
        energy = np.append(energy_lo.value,
                           energy_hi[-1].value) * energy_lo.unit
        data = table['{}'.format(data_col)].quantity
        return cls(energy=energy, data=data)

    def plot(self, ax=None, energy=None, **kwargs):
        """Plot background rate

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy : `~astropy.units.Quantity`
            Energy nodes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis

        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        if energy is None:
            energy = self.energy.nodes
        values = self.evaluate(energy=energy)
        xerr = (energy.value - self.energy.data[:-1].value,
                self.energy.data[1:].value - energy.value)
        ax.errorbar(energy.value, values.value, xerr=xerr, fmt='o', **kwargs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy [{}]'.format(self.energy.unit))
        ax.set_ylabel('Background rate [{}]'.format(self.data.unit))

        return ax


class Psf68Table(NDDataArray):
    """Background rate Table

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.
    Work will be done to fix this.

    Parameters
    -----------
    energy : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
        Bin edges of energy axis
    data : `~astropy.units.Quantity`
        Background rate
    """
    energy = BinnedDataAxis(interpolation_mode='log')
    """Energy Axis"""
    axis_names = ['energy']

    @classmethod
    def from_table(cls, table):
        """PSF reader"""
        energy_col = 'ENERG'
        data_col = 'PSF68'

        energy_lo = table['{}_LO'.format(energy_col)].quantity
        energy_hi = table['{}_HI'.format(energy_col)].quantity
        energy = np.append(
            energy_lo.value, energy_hi[-1].value) * energy_lo.unit
        data = table['{}'.format(data_col)].quantity
        return cls(energy=energy, data=data)

    def plot(self, ax=None, energy=None, **kwargs):
        """Plot point spread function

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        energy : `~astropy.units.Quantity`
            Energy nodes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis

        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        if energy is None:
            energy = self.energy.nodes
        values = self.evaluate(energy=energy)
        xerr = (energy.value - self.energy.data[:-1].value,
                self.energy.data[1:].value - energy.value)
        ax.errorbar(energy.value, values.value, xerr=xerr, fmt='o', **kwargs)
        ax.set_xscale('log')
        ax.set_xlabel('Energy [{}]'.format(self.energy.unit))
        ax.set_ylabel(
            'Angular resolution 68 % containment [{}]'.format(self.data.unit)
        )

        return ax


class CTAPerf(object):
    """CTA instrument response function container.

    Class handling CTA performance.

    For now we use the production 2 of the CTA IRF
    (https://portal.cta-observatory.org/Pages/CTA-Performance.aspx)

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.
    Work will be done to handle better the PSF and the background rate.

    This class is similar to `~gammapy.data.DataStoreObservation`,
    but only contains performance (no event data or livetime info).
    TODO: maybe re-factor code somehow to avoid code duplication.

    Parameters
    ----------
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    psf : `~gammapy.scripts.Psf68Table`
        Point spread function
    bkg : `~gammapy.scripts.BgRateTable`
        Background rate
    """

    def __init__(self, aeff=None, edisp=None, psf=None, bkg=None):
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg

    @classmethod
    def read(cls, filename):
        """
        Read from a FITS file.

        Parameters
        ----------
        filename : `str`
            File containing the IRFs
        """
        filename = str(make_path(filename))

        aeff = EffectiveAreaTable.read(filename, hdu='SPECRESP')

        # Get EnergyDispersion from HDUList
        hdu_list = fits.open(filename)
        edisp = EnergyDispersion.from_hdulist(hdu_list=hdu_list)

        # Do not work, can't understand why, see BgRateTable and Psf68Table class
        # bkg = BgRateTable.read(filename, hdu='BACKGROUND')
        bkg = BgRateTable.from_hdulist(
            fits.HDUList([hdu_list[0], hdu_list['BACKGROUND']])
        )

        psf = Psf68Table.from_hdulist(
            fits.HDUList([hdu_list[0], hdu_list['POINT SPREAD FUNCTION']])
        )

        return cls(
            aeff=aeff,
            bkg=bkg,
            edisp=edisp,
            psf=psf,
        )

    def peek(self, figsize=(10, 10)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        self.bkg.plot(ax=axes[0][0])
        aeff = self.aeff.plot(ax=axes[0][1])
        aeff.set_yscale('log')

        self.psf.plot(ax=axes[1][0])

        self.edisp.plot_matrix(ax=axes[1][1])

        plt.tight_layout()
        plt.show()
        return fig
