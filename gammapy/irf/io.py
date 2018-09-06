# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.io import fits
from astropy.table import Table
from ..utils.scripts import make_path
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.energy import EnergyBounds
from .effective_area import EffectiveAreaTable2D, EffectiveAreaTable
from .background import Background3D
from .energy_dispersion import EnergyDispersion2D
from .psf_gauss import EnergyDependentMultiGaussPSF

__all__ = ["CTAIrf", "BgRateTable", "Psf68Table", "SensitivityTable", "CTAPerf"]


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
    bkg : `~gammapy.irf.Background3D`
        Background rate
    ref_sensi : `~gammapy.irf.SensitivityTable`
        Reference Sensitivity
    """

    def __init__(self, aeff=None, edisp=None, psf=None, bkg=None, ref_sensi=None):
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg
        self.ref_sensi = ref_sensi

    @classmethod
    def read(cls, filename):
        """Read from a FITS file.

        Parameters
        ----------
        filename : str
            File containing the IRFs
        """
        filename = str(make_path(filename))
        hdu_list = fits.open(filename)

        aeff = EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")
        bkg = Background3D.read(filename, hdu="BACKGROUND")
        edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu="POINT SPREAD FUNCTION")

        if "SENSITIVITY" in hdu_list:
            sensi = SensitivityTable.read(filename, hdu="SENSITIVITY")
        else:
            sensi = None

        return cls(aeff=aeff, bkg=bkg, edisp=edisp, psf=psf, ref_sensi=sensi)


class BgRateTable(object):
    """Background rate table.

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.
    Work will be done to fix this.

    Parameters
    -----------
    energy_lo, energy_hi : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
        Bin edges of energy axis
    data : `~astropy.units.Quantity`
        Background rate
    """

    def __init__(self, energy_lo, energy_hi, data):
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            )
        ]
        self.data = NDDataArray(axes=axes, data=data)

    @property
    def energy(self):
        return self.data.axes[0]

    @classmethod
    def from_table(cls, table):
        """Background rate reader"""
        energy_lo = table["ENERG_LO"].quantity
        energy_hi = table["ENERG_HI"].quantity
        data = table["BGD"].quantity
        return cls(energy_lo=energy_lo, energy_hi=energy_hi, data=data)

    @classmethod
    def from_hdulist(cls, hdulist, hdu="BACKGROUND"):
        fits_table = hdulist[hdu]
        table = Table.read(fits_table)
        return cls.from_table(table)

    @classmethod
    def read(cls, filename, hdu="BACKGROUND"):
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    def plot(self, ax=None, energy=None, **kwargs):
        """Plot background rate.

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

        energy = energy or self.energy.nodes
        values = self.data.evaluate(energy=energy)
        xerr = (
            energy.value - self.energy.lo.value,
            self.energy.hi.value - energy.value,
        )
        ax.errorbar(energy.value, values.value, xerr=xerr, fmt="o", **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Energy [{}]".format(self.energy.unit))
        ax.set_ylabel("Background rate [{}]".format(self.data.data.unit))

        return ax


class Psf68Table(object):
    """Background rate table.

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.
    Work will be done to fix this.

    Parameters
    -----------
    energy_lo, energy_hi : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
        Bin edges of energy axis
    data : `~astropy.units.Quantity`
        Background rate
    """

    def __init__(self, energy_lo, energy_hi, data):
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            )
        ]
        self.data = NDDataArray(axes=axes, data=data)

    @property
    def energy(self):
        return self.data.axes[0]

    @classmethod
    def from_table(cls, table):
        """PSF reader"""
        energy_lo = table["ENERG_LO"].quantity
        energy_hi = table["ENERG_HI"].quantity
        data = table["PSF68"].quantity
        return cls(energy_lo=energy_lo, energy_hi=energy_hi, data=data)

    @classmethod
    def from_hdulist(cls, hdulist, hdu="POINT SPREAD FUNCTION"):
        fits_table = hdulist[hdu]
        table = Table.read(fits_table)
        return cls.from_table(table)

    @classmethod
    def read(cls, filename, hdu="POINT SPREAD FUNCTION"):
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    def plot(self, ax=None, energy=None, **kwargs):
        """Plot point spread function.

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

        energy = energy or self.energy.nodes
        values = self.data.evaluate(energy=energy)
        xerr = (
            energy.value - self.energy.lo.value,
            self.energy.hi.value - energy.value,
        )
        ax.errorbar(energy.value, values.value, xerr=xerr, fmt="o", **kwargs)
        ax.set_xscale("log")
        ax.set_xlabel("Energy [{}]".format(self.energy.unit))
        ax.set_ylabel(
            "Angular resolution 68 % containment [{}]".format(self.data.data.unit)
        )

        return ax


class SensitivityTable(object):
    """Sensitivity table.

    The IRF format should be compliant with the one discussed
    at http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/.
    Work will be done to fix this.

    Parameters
    -----------
    energy_lo, energy_hi : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
        Bin edges of energy axis
    data : `~astropy.units.Quantity`
        Sensitivity
    """

    def __init__(self, energy_lo, energy_hi, data):
        axes = [
            BinnedDataAxis(
                energy_lo, energy_hi, interpolation_mode="log", name="energy"
            )
        ]
        self.data = NDDataArray(axes=axes, data=data)

    @property
    def energy(self):
        return self.data.axis("energy")

    @classmethod
    def from_table(cls, table):
        energy_lo = table["ENERG_LO"].quantity
        energy_hi = table["ENERG_HI"].quantity
        data = table["SENSITIVITY"].quantity
        return cls(energy_lo=energy_lo, energy_hi=energy_hi, data=data)

    @classmethod
    def from_hdulist(cls, hdulist, hdu="SENSITIVITY"):
        fits_table = hdulist[hdu]
        table = Table.read(fits_table)
        return cls.from_table(table)

    @classmethod
    def read(cls, filename, hdu="SENSITVITY"):
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu=hdu)

    def plot(self, ax=None, energy=None, **kwargs):
        """Plot sensitivity.

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

        energy = energy or self.energy.nodes
        values = self.data.evaluate(energy=energy)
        xerr = (
            energy.value - self.energy.lo.value,
            self.energy.hi.value - energy.value,
        )
        ax.errorbar(energy.value, values.value, xerr=xerr, fmt="o", **kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reco Energy [{}]".format(self.energy.unit))
        ax.set_ylabel("Sensitivity [{}]".format(self.data.data.unit))

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
    edisp : `~gammapy.irf.EnergyDispersion2D`
        Energy dispersion
    psf : `~gammapy.scripts.Psf68Table`
        Point spread function
    bkg : `~gammapy.scripts.BgRateTable`
        Background rate
    sens : `~gammapy.scripts.SensitivityTable`
        Sensitivity
    rmf: `~gammapy.irf.EnergyDispersion`
        RMF
    """

    def __init__(self, aeff=None, edisp=None, psf=None, bkg=None, sens=None, rmf=None):
        self.aeff = aeff
        self.edisp = edisp
        self.psf = psf
        self.bkg = bkg
        self.sens = sens
        self.rmf = rmf

    @classmethod
    def read(cls, filename, offset="0.5 deg"):
        """Read from a FITS file.

        Compute RMF at 0.5 deg offset on fly.

        Parameters
        ----------
        filename : str
            File containing the IRFs
        """
        filename = str(make_path(filename))

        with fits.open(filename, memmap=False) as hdulist:
            aeff = EffectiveAreaTable.from_hdulist(hdulist=hdulist)
            edisp = EnergyDispersion2D.read(filename, hdu="ENERGY DISPERSION")
            bkg = BgRateTable.from_hdulist(hdulist=hdulist)
            psf = Psf68Table.from_hdulist(hdulist=hdulist)
            sens = SensitivityTable.from_hdulist(hdulist=hdulist)

        # Create rmf with appropriate dimensions (e_reco->bkg, e_true->area)
        e_reco_min = bkg.energy.lo[0]
        e_reco_max = bkg.energy.hi[-1]
        e_reco_bin = bkg.energy.nbins
        e_reco_axis = EnergyBounds.equal_log_spacing(
            e_reco_min, e_reco_max, e_reco_bin, "TeV"
        )

        e_true_min = aeff.energy.lo[0]
        e_true_max = aeff.energy.hi[-1]
        e_true_bin = aeff.energy.nbins
        e_true_axis = EnergyBounds.equal_log_spacing(
            e_true_min, e_true_max, e_true_bin, "TeV"
        )

        rmf = edisp.to_energy_dispersion(
            offset=offset, e_reco=e_reco_axis, e_true=e_true_axis
        )

        return cls(aeff=aeff, bkg=bkg, edisp=edisp, psf=psf, sens=sens, rmf=rmf)

    def peek(self, figsize=(15, 8)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)
        ax_bkg = plt.subplot2grid((2, 4), (0, 0))
        ax_area = plt.subplot2grid((2, 4), (0, 1))
        ax_sens = plt.subplot2grid((2, 4), (0, 2), colspan=2, rowspan=2)
        ax_psf = plt.subplot2grid((2, 4), (1, 0))
        ax_resol = plt.subplot2grid((2, 4), (1, 1))

        self.bkg.plot(ax=ax_bkg)
        self.aeff.plot(ax=ax_area).set_yscale("log")
        self.sens.plot(ax=ax_sens)
        self.psf.plot(ax=ax_psf)
        self.edisp.plot_bias(ax=ax_resol, offset="0.5 deg")

        ax_bkg.grid(which="both")
        ax_area.grid(which="both")
        ax_sens.grid(which="both")
        ax_psf.grid(which="both")
        fig.tight_layout()

    @staticmethod
    def superpose_perf(cta_perf, labels):
        """Superpose performance plot.

        Parameters
        ----------
        cta_perf : list of `~gammapy.scripts.CTAPerf`
           List of performance
        labels : list of str
           List of labels
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax_bkg = plt.subplot2grid((2, 2), (0, 0))
        ax_area = plt.subplot2grid((2, 2), (0, 1))
        ax_psf = plt.subplot2grid((2, 2), (1, 0))
        ax_sens = plt.subplot2grid((2, 2), (1, 1))

        for index, (perf, label) in enumerate(zip(cta_perf, labels)):
            plot_label = {"label": label}
            perf.bkg.plot(ax=ax_bkg, **plot_label)
            perf.aeff.plot(ax=ax_area, **plot_label).set_yscale("log")
            perf.sens.plot(ax=ax_sens, **plot_label)
            perf.psf.plot(ax=ax_psf, **plot_label)

        ax_bkg.legend(loc="best")
        ax_area.legend(loc="best")
        ax_psf.legend(loc="best")
        ax_sens.legend(loc="best")

        ax_bkg.grid(which="both")
        ax_area.grid(which="both")
        ax_psf.grid(which="both")
        ax_sens.grid(which="both")

        fig.tight_layout()
