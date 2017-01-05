# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.table import Table
from ..utils.energy import EnergyBounds, Energy
from ..utils.array import array_stats_str
from ..utils.scripts import make_path
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.fits import energy_axis_to_ebounds

__all__ = [
    'EnergyDispersion',
    'EnergyDispersion2D',
]


class EnergyDispersion(object):
    """Energy dispersion matrix.

    We use a dense matrix (`numpy.ndarray`) for the energy dispersion matrix.
    An alternative would be to store a sparse matrix
    (`scipy.sparse.csc_matrix`).  It's not clear which would be more efficient
    for typical gamma-ray energy dispersion matrices.

    The most common file format for energy dispersion matrices is the RMF
    (Redistribution Matrix File) format from X-ray astronomy:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html

    Parameters
    ----------
    e_true : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
        Bin edges of true energy axis
    e_reco : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
        Bin edges of reconstruced energy axis
    data : array_like
        2-dim energy dispersion matrix (probability density).
    """
    default_interp_kwargs = dict(bounds_error=False, fill_value=0)
    """Default Interpolation kwargs for `~NDDataArray`"""

    def __init__(self, e_true, e_reco, data, interp_kwargs=None, meta=None):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs
        axes = [
            BinnedDataAxis(e_true, interpolation_mode='log', name='e_true'),
            BinnedDataAxis(e_reco, interpolation_mode='log', name='e_reco')
        ]
        self.data=NDDataArray(axes=axes, data=data,
                                interp_kwargs=interp_kwargs)
        if meta is not None:
            self.meta=Bunch(meta)

    @property
    def pdf_matrix(self):
        """PDF matrix `~numpy.ndarray`

        Rows (first index): True Energy
        Columns (second index): Reco Energy
        """
        return self.data.data.value

    def pdf_in_safe_range(self, lo_threshold, hi_threshold):
        """PDF matrix with bins outside threshold set to 0

        Parameters
        ----------
        lo_threshold : `~astropy.units.Quantity`
            Low reco energy threshold
        hi_threshold : `~astropy.units.Quantity`
            High reco energy threshold
        """
        data=self.pdf_matrix.copy()
        idx=np.where((self.data.axis('e_reco').data[:-1] < lo_threshold) |
                       (self.data.axis('e_reco').data[1:] > hi_threshold))
        data[:, idx]=0
        return data

    @classmethod
    def from_gauss(cls, e_true, e_reco, sigma=0.2, pdf_threshold=1e-6):
        """Create Gaussian `EnergyDispersion` matrix.

        The output matrix will be Gaussian in log(e_true / e_reco)

        TODO: extend to have a vector of bias various true energies.
        TODO: extend to have vector of  resolution for various true energies.
        TODO: give formula: Gaussian in log(e_reco)
        TODO: add option to add poisson noise

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
            Bin edges of true energy axis
        e_reco : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
            Bin edges of reconstructed energy axis
        sigma : float, optional
            RMS width of Gaussian energy dispersion, resolution
        pdf_threshold : float, optional
            Zero suppression threshold
        """
        from scipy.special import erf

        e_reco = EnergyBounds(e_reco)
        e_true = EnergyBounds(e_true)

        # erf does not work with Quantities
        reco=e_reco.to('TeV').value
        true=e_true.log_centers.to('TeV').value
        migra_min=np.log10(reco[:-1] / true[:, np.newaxis])
        migra_max=np.log10(reco[1:] / true[:, np.newaxis])

        pdf=.5 * (erf(migra_max / (np.sqrt(2.) * sigma))
                    - erf(migra_min / (np.sqrt(2.) * sigma)))

        pdf[np.where(pdf < pdf_threshold)]=0

        return cls(e_true=e_true, e_reco=e_reco, data=pdf)

    @classmethod
    def from_hdulist(cls, hdulist, hdu1='MATRIX', hdu2='EBOUNDS'):
        """Create `EnergyDispersion` object from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list with ``MATRIX`` and ``EBOUNDS`` extensions.
        hdu1 : str, optional
            HDU containing the energy dispersion matrix, default: MATRIX
        hdu2 : str, optional
            HDU containing the energy axis information, default, EBOUNDS
        """
        matrix_hdu = hdulist[hdu1]
        ebounds_hdu = hdulist[hdu2]

        data=matrix_hdu.data
        header=matrix_hdu.header

        pdf_matrix=np.zeros([len(data), header['DETCHANS']], dtype=np.float64)

        for i, l in enumerate(data):
            if l.field('N_GRP'):
                m_start=0
                for k in range(l.field('N_GRP')):
                    pdf_matrix[i, l.field('F_CHAN')[k]: l.field(
                        'F_CHAN')[k] + l.field('N_CHAN')[k]]=l.field(
                        'MATRIX')[m_start:m_start + l.field('N_CHAN')[k]]
                    m_start += l.field('N_CHAN')[k]

        e_reco=EnergyBounds.from_ebounds(ebounds_hdu)
        e_true=EnergyBounds.from_rmf_matrix(matrix_hdu)

        return cls(data=pdf_matrix, e_true=e_true, e_reco=e_reco)

    @classmethod
    def read(cls, filename, hdu1='MATRIX', hdu2='EBOUNDS', **kwargs):
        """Read from file

        Parameters
        ----------
        filename : `~gammapy.extern.pathlib.Path`, str
            File to read
        hdu1 : str, optional
            HDU containing the energy dispersion matrix, default: MATRIX
        hdu2 : str, optional
            HDU containing the energy axis information, default, EBOUNDS
        """
        filename = make_path(filename)
        hdulist = fits.open(str(filename), **kwargs)
        try:
            return cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2)
        except KeyError:
            msg = 'File {} contains no HDU "{}"'.format(filename, hdu)
            msg += '\n Available {}'.format([_.name for _ in hdulist])
            raise ValueError(msg)

    def to_hdulist(self, **kwargs):
        """
        Convert RM to FITS HDU list format.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            Header to be written in the fits file.
        energy_unit : str
            Unit in which the energy is written in the HDU list

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            RMF in HDU list format.

        Notes
        -----
        For more info on the RMF FITS file format see:
        http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html

        """
        # Cannot use table_to_fits here due to variable length array
        # http://docs.astropy.org/en/v1.0.4/io/fits/usage/unfamiliar.html

        table=self.to_table()
        name=table.meta.pop('name')

        header=fits.Header()
        header.update(table.meta)

        cols=table.columns
        c0=fits.Column(name=cols[0].name, format='E', array=cols[0],
                         unit='{}'.format(cols[0].unit))
        c1=fits.Column(name=cols[1].name, format='E', array=cols[1],
                         unit='{}'.format(cols[1].unit))
        c2=fits.Column(name=cols[2].name, format='I', array=cols[2])
        c3=fits.Column(name=cols[3].name, format='PI()', array=cols[3])
        c4=fits.Column(name=cols[4].name, format='PI()', array=cols[4])
        c5=fits.Column(name=cols[5].name, format='PE()', array=cols[5])

        hdu=fits.BinTableHDU.from_columns([c0, c1, c2, c3, c4, c5],
                                            header=header, name=name)

        ebounds=energy_axis_to_ebounds(self.data.axis('e_reco'))
        prim_hdu=fits.PrimaryHDU()

        return fits.HDUList([prim_hdu, hdu, ebounds])

    def to_table(self):
        """Convert to `~astropy.table.Table`.

        The output table is in the OGIP RMF format.
        http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#Tab:1
        """
        table=Table()

        rows=self.pdf_matrix.shape[0]
        n_grp=[]
        f_chan=np.ndarray(dtype=np.object, shape=rows)
        n_chan=np.ndarray(dtype=np.object, shape=rows)
        matrix=np.ndarray(dtype=np.object, shape=rows)

        # Make RMF type matrix
        for i, row in enumerate(self.data.data.value):
            subsets=1
            pos=np.nonzero(row)[0]
            borders=np.where(np.diff(pos) != 1)[0]
            # add 1 to borders for correct behaviour of np.split
            groups=np.asarray(np.split(pos, borders + 1))
            n_grp_temp=groups.shape[0] if groups.size > 0 else 1
            n_chan_temp=np.asarray([val.size for val in groups])
            try:
                f_chan_temp=np.asarray([val[0] for val in groups])
            except(IndexError):
                f_chan_temp=np.zeros(1)

            n_grp.append(n_grp_temp)
            f_chan[i]=f_chan_temp
            n_chan[i]=n_chan_temp
            matrix[i]=row[pos]

        table['ENERG_LO']=self.data.axis('e_true').data[:-1]
        table['ENERG_HI']=self.data.axis('e_true').data[1:]
        table['N_GRP']=np.asarray(n_grp, dtype=np.int16)
        table['F_CHAN']=f_chan
        table['N_CHAN']=n_chan
        table['MATRIX']=matrix

        # Get total number of groups and channel subsets
        numgrp, numelt=0, 0
        for val, val2 in zip(table['N_GRP'], table['N_CHAN']):
            numgrp += np.sum(val)
            numelt += np.sum(val2)

        meta=dict(name='MATRIX',
                    chantype='PHA',
                    hduclass='OGIP',
                    hduclas1='RESPONSE',
                    hduclas2='RSP_MATRIX',
                    detchans=self.data.axis('e_reco').nbins,
                    numgrp=numgrp,
                    numelt=numelt,
                    tlmin4=0,
                    )

        table.meta=meta
        return table

    def write(self, filename, **kwargs):
        filename = make_path(filename)
        self.to_hdulist().writeto(str(filename), **kwargs)

    def get_resolution(self, e_true):
        """Get energy resolution for a fiven true energy

        Resolution is the 1 sigma containment of the energy dispersion PDF.

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`
            True energy
        """
        # Variance is 2nd moment of PDF
        pdf=self.data.evaluate(e_true=e_true)
        mean=self._get_mean(e_true)
        temp=(self.data.axis('e_reco')._interp_nodes() - mean) ** 2
        var=np.sum(temp * pdf)
        return np.sqrt(var)

    def get_bias(self, e_true):
        r"""Get reconstruction bias for a given true energy

        Bias is defined as

        .. math::

            \frac{E_{reco}-E_{true}}{E_{true}}

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`
            True energy
        """
        mean=self._get_mean(e_true)
        e_reco=(10 ** mean) * self.data.axis('e_reco').unit
        bias=(e_true - e_reco) / e_true
        return bias

    def _get_mean(self, e_true):
        r"""Get mean log reconstructed energy
        """
        # Reconstructed energy is 1st moment of PDF
        pdf=self.data.evaluate(e_true=e_true)
        norm=np.sum(pdf)
        temp=np.sum(pdf * self.data.axis('e_reco')._interp_nodes())
        return temp / norm

    def apply(self, data, e_reco=None):
        """Apply energy dispersion.

        Computes the matrix product of ``data``
        (which typically is model flux or counts in true energy bins)
        with the energy dispersion matrix.

        Parameters
        ----------
        data : array_like
            1-dim data array.
        e_reco : `~astropy.units.Quantity`, optional
            Desired energy binning of the convolved data, if provided the
            `~gammapy.irf.EnergyDispersion` is evaluated at the log centers of
            the energy axis.

        Returns
        -------
        convolved_data : array
            1-dim data array after multiplication with the energy dispersion matrix
        """
        if e_reco is None:
            e_reco=self.data.axis('e_reco').nodes
        else:
            e_reco=np.sqrt(e_reco[:-1] * e_reco[1:])
        edisp_pdf=self.data.evaluate(e_reco=e_reco)
        return np.dot(data, edisp_pdf)

    def _extent(self):
        """Extent (x0, x1, y0, y1) for plotting (4x float)

        x stands for true energy and y for reconstructed energy
        """
        x=self.data.axis('e_true').data[[0, -1]].value
        y=self.data.axis('e_reco').data[[0, -1]].value
        return x[0], x[1], y[0], y[1]

    def plot_matrix(self, ax=None, show_energy=None, **kwargs):
        """Plot PDF matrix.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        show_energy : `~astropy.units.Quantity`, optional
            Show energy, e.g. threshold, as vertical line

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        kwargs.setdefault('cmap', 'afmhot')
        kwargs.setdefault('origin', 'bottom')
        kwargs.setdefault('interpolation', 'nearest')
        kwargs.setdefault('norm', PowerNorm(gamma=0.5))

        ax=plt.gca() if ax is None else ax

        image=self.pdf_matrix.transpose()
        ax.imshow(image, extent=self._extent(), **kwargs)
        if show_energy is not None:
            ener_val=Quantity(show_energy).to(self.reco_energy.unit).value
            ax.hlines(ener_val, 0, 200200, linestyles='dashed')

        ax.set_xlabel('True energy (TeV)')
        ax.set_ylabel('Reco energy (TeV)')

        ax.set_xscale('log')
        ax.set_yscale('log')

        return ax

    def plot_bias(self, ax=None, **kwargs):
        """Plot reconstruction bias.

        see :func:`~gammapy.irf.EnergyDispersion.get_bias`

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        ax=plt.gca() if ax is None else ax

        x=self.data.axis('e_true').nodes.to('TeV').value
        y=self.get_bias(self.data.axis('e_true').nodes)

        ax.plot(x, y, **kwargs)
        ax.set_xlabel('True energy [TeV]')
        ax.set_ylabel(r'(E_{true} - E_{reco} / E_{true})')
        ax.set_xscale('log')
        return ax

    def to_sherpa(self, name):
        """Return `~sherpa.astro.data.DataARF`

        Parameters
        ----------
        name : str
            Instance name
        """
        from sherpa.astro.data import DataRMF
        from sherpa.utils import SherpaInt, SherpaUInt, SherpaFloat

        # Need to modify RMF data
        # see https://github.com/sherpa/sherpa/blob/master/sherpa/astro/io/pyfits_backend.py#L727

        table=self.to_table()
        n_grp=table['N_GRP'].data.astype(SherpaUInt)
        f_chan=table['F_CHAN'].data
        f_chan=np.concatenate([row for row in f_chan]).astype(SherpaUInt)
        n_chan=table['N_CHAN'].data
        n_chan=np.concatenate([row for row in n_chan]).astype(SherpaUInt)
        matrix=table['MATRIX'].data

        good=n_grp > 0
        matrix=matrix[good]
        matrix=np.concatenate([row for row in matrix])
        matrix=matrix.astype(SherpaFloat)

        # TODO: Not sure if we need this if statement
        if f_chan.ndim > 1 and n_chan.ndim > 1:
            f_chan=[]
            n_chan=[]
            for grp, fch, nch, in izip(n_grp, f_chan, n_chan):
                for i in xrange(grp):
                    f_chan.append(fch[i])
                    n_chan.append(nch[i])

            f_chan=numpy.asarray(f_chan, SherpaUInt)
            n_chan=numpy.asarray(n_chan, SherpaUInt)
        else:
            if len(n_grp) == len(f_chan):
                good=n_grp > 0
                f_chan=f_chan[good]
                n_chan=n_chan[good]

        kwargs=dict(
            name=name,
            energ_lo=table['ENERG_LO'].quantity.to('keV').value.astype(SherpaFloat),
            energ_hi=table['ENERG_HI'].quantity.to('keV').value.astype(SherpaFloat),
            matrix=matrix,
            n_grp=n_grp,
            n_chan=n_chan,
            f_chan=f_chan,
            detchans=self.data.axis('e_reco').nbins,
            e_min=self.data.axis('e_reco').data[:-1].to('keV').value,
            e_max=self.data.axis('e_reco').data[1:].to('keV').value,
            offset=0,
        )

        return DataRMF(**kwargs)


class EnergyDispersion2D(object):
    """Offset-dependent energy dispersion matrix.

    Parameters
    ----------
    etrue_lo : `~gammapy.utils.energy.Energy`
        True energy lower bounds
    etrue_hi : `~gammapy.utils.energy.Energy`
        True energy upper bounds
    migra_lo : `~numpy.ndarray`, list
        Migration lower bounds
    migra_hi : `~numpy.ndarray`, list
        Migration upper bounds
    offset_lo : `~astropy.coordinates.Angle`
        Offset lower bounds
    offset_hi : `~astropy.coordinates.Angle`
        Offset lower bounds
    dispersion : `~numpy.ndarray`
        PDF matrix
    interp_kwargs : dict or None
        Interpolation parameter dict passed to `scipy.interpolate.RegularGridInterpolator`.
        If you pass ``None``, the default ``interp_params=dict(bounds_error=False, fill_value=0)`` is used.

    Examples
    --------

    Plot migration histogram for a given offset and true energy

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDispersion2D
        filename = '$GAMMAPY_EXTRA/test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz'
        edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
        edisp.plot_migration()
        plt.xlim(0, 4)


    Plot evolution of bias and resolution as a function of true energy
    for a given offset

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from gammapy.irf import EnergyDispersion2D
        from gammapy.utils.energy import Energy
        from astropy.coordinates import Angle
        filename = '$GAMMAPY_EXTRA/test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz'
        edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
        migra = np.linspace(0.1,2,80)
        e_true = Energy.equal_log_spacing(0.13,60,60,'TeV')
        offset = Angle([0.554], 'deg')
        edisp.plot_bias(offset=offset, e_true=e_true, migra=migra)
        plt.xscale('log')

    Create RMF matrix

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDispersion2D
        from gammapy.utils.energy import EnergyBounds
        filename = '$GAMMAPY_EXTRA/test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz'
        edisp = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
        e_axis = EnergyBounds.equal_log_spacing(0.1,20,60, 'TeV')
        rmf = edisp.to_energy_dispersion('1.2 deg', e_reco = e_axis, e_true = e_axis)
        rmf.plot_matrix()
        plt.loglog()

    """

    def __init__(self, etrue_lo, etrue_hi, migra_lo, migra_hi, offset_lo,
                 offset_hi, dispersion, interp_kwargs=None):

        if not isinstance(etrue_lo, Quantity) or not isinstance(etrue_hi, Quantity):
            raise ValueError("Energies must be Quantity objects.")
        if not isinstance(offset_lo, Angle) or not isinstance(offset_hi, Angle):
            raise ValueError("Offsets must be Angle objects.")

        self.migra_lo=migra_lo
        self.migra_hi=migra_hi
        self.offset_lo=offset_lo
        self.offset_hi=offset_hi
        self.dispersion=dispersion

        self.ebounds=EnergyBounds.from_lower_and_upper_bounds(etrue_lo, etrue_hi)
        self.energy=self.ebounds.log_centers
        self.offset=(offset_hi + offset_lo) / 2
        self.migra=(migra_hi + migra_lo) / 2

        if not interp_kwargs:
            interp_kwargs=dict(bounds_error=False, fill_value=0)

        self._prepare_linear_interpolator(interp_kwargs)

    @classmethod
    def from_fits(cls, hdu):
        """Create from a FITS HDU.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            ``ENERGY DISPERSION`` extension.

        """
        data=hdu.data
        header=hdu.header
        e_lo=EnergyBounds(data['ETRUE_LO'].squeeze(), header['TUNIT1'])
        e_hi=EnergyBounds(data['ETRUE_HI'].squeeze(), header['TUNIT2'])
        o_lo=Angle(data['THETA_LO'].squeeze(), header['TUNIT5'])
        o_hi=Angle(data['THETA_HI'].squeeze(), header['TUNIT6'])
        m_lo=data['MIGRA_LO'].squeeze()
        m_hi=data['MIGRA_HI'].squeeze()
        matrix=data['MATRIX'].squeeze()

        return cls(e_lo, e_hi, m_lo, m_hi, o_lo, o_hi, matrix)

    @classmethod
    def read(cls, filename, hdu='edisp_2d'):
        """Read from FITS file.

        See :ref:`gadf:edisp_2d`

        Parameters
        ----------
        filename : str
            File name
        """
        filename=make_path(filename)
        hdulist=fits.open(str(filename))
        hdu=hdulist[hdu]
        return cls.from_fits(hdu)

    def evaluate(self, offset=None, e_true=None, migra=None):
        """Probability for a given offset, true energy, and migration

        Parameters
        ----------
        e_true : `~gammapy.utils.energy.Energy`, optional
            True energy
        migra : `~numpy.ndarray`, optional
            Energy migration e_reco/e_true
        offset : `~astropy.coordinates.Angle`, optional
            Offset
        """

        offset=self.offset if offset is None else Angle(offset)
        e_true=self.energy if e_true is None else Energy(e_true)
        migra=self.migra if migra is None else migra

        offset=offset.to('deg')
        e_true=e_true.to('TeV')

        val=self._eval(offset=offset, e_true=e_true, migra=migra)

        return val

    def _eval(self, offset=None, e_true=None, migra=None):

        x=np.atleast_1d(offset.value)
        y=np.atleast_1d(migra)
        z=np.atleast_1d(np.log10(e_true.value))
        in_shape=(x.size, y.size, z.size)

        pts=[[xx, yy, zz] for xx in x for yy in y for zz in z]
        val_array=self._linear(pts)

        return val_array.reshape(in_shape).squeeze()

    def to_energy_dispersion(self, offset, e_true=None, e_reco=None):
        """Detector response R(Delta E_reco, Delta E_true)

        Probability to reconstruct an energy in a given true energy band
        in a given reconstructed energy band

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset
        e_true : `~gammapy.utils.energy.EnergyBounds`, None
            True energy axis
        e_reco : `~gammapy.utils.energy.EnergyBounds`
            Reconstructed energy axis

        Returns
        -------
        edisp : `~gammapy.irf.EnergyDispersion`
            Energy disperion matrix
        """
        offset=Angle(offset)
        e_true=self.ebounds if e_true is None else EnergyBounds(e_true)
        e_reco=self.ebounds if e_reco is None else EnergyBounds(e_reco)

        rm=[]

        for energy in e_true.log_centers:
            vec=self.get_response(offset=offset, e_true=energy, e_reco=e_reco)
            rm.append(vec)

        rm=np.asarray(rm)
        return EnergyDispersion(data=rm, e_true=e_true, e_reco=e_reco)

    def get_response(self, offset, e_true, e_reco=None):
        """Detector response R(Delta E_reco, E_true)

        Probability to reconstruct a given true energy in a given reconstructed
        energy band. In each reco bin, you integrate with a riemann sum over the default migra bin of your analysis.

        Parameters
        ----------
        e_true : `~gammapy.utils.energy.Energy`
            True energy
        e_reco : `~gammapy.utils.energy.EnergyBounds`, None
            Reconstructed energy axis
        offset : `~astropy.coordinates.Angle`
            Offset

        Returns
        -------
        rv : `~numpy.ndarray`
            Redistribution vector
        """

        e_true=Energy(e_true)

        # Default: e_reco nodes = migra nodes * e_true nodes
        if e_reco is None:
            e_reco=EnergyBounds.from_lower_and_upper_bounds(
                self.migra_lo * e_true, self.migra_hi * e_true)
            migra=self.migra

        # Translate given e_reco binning to migra at bin center
        else:
            e_reco=EnergyBounds(e_reco)
            center=e_reco.log_centers
            migra=center / e_true

        # ensure to have a normalized edisp
        migra_bin=self.migra_hi - self.migra_lo
        if (migra_bin[0] == migra_bin[1]):
            migra_mean=1 / 2. * (self.migra_hi + self.migra_lo)
        else:
            migra_mean=np.sqrt(self.migra_hi * self.migra_lo)
        val_norm=self.evaluate(offset=offset, e_true=e_true, migra=migra_mean)
        norm=np.sum(val_norm * migra_bin)

        migra_e_reco=e_reco / e_true
        integral=np.zeros(len(e_reco) - 1)
        if norm != 0:
            for i, migra_int in enumerate(migra_e_reco[:-1]):
                # The migra_e_reco bin is inferior to the migra min in the dispersion fits file
                if migra_e_reco[i + 1] < migra_mean[0]:
                    continue
                # The migra_e_reco bin is superior to the migra max in the dispersion fits file
                elif migra_e_reco[i] > migra_mean[-1]:
                    continue
                else:
                    if migra_e_reco[i] < migra_mean[0]:
                        i_min=0
                    else:
                        i_min=np.where(migra_mean < migra_e_reco[i])[0][-1]
                    i_max=np.where(migra_mean < migra_e_reco[i + 1])[0][-1]

                    migra_bin_reco=migra_bin[i_min + 1:i_max]
                    if migra_e_reco[i + 1] > migra_mean[-1]:
                        index=np.arange(i_min, i_max, 1)
                        migra_bin_reco=np.insert(migra_bin_reco, 0, (migra_mean[i_min + 1] - migra_e_reco[i]))
                    elif migra_e_reco[i] < migra_mean[0]:
                        if i_max == 0:
                            index=np.arange(i_min, i_max + 1, 1)
                        else:
                            index=np.arange(i_min + 1, i_max + 1, 1)
                        migra_bin_reco=np.append(migra_bin_reco, migra_e_reco[i + 1] - migra_mean[i_max])
                    elif i_min == i_max:
                        index=np.arange(i_min, i_max + 1, 1)
                        migra_bin_reco=np.insert(migra_bin_reco, 0, (migra_e_reco[i + 1] - migra_e_reco[i]))
                    else:
                        index=np.arange(i_min, i_max + 1, 1)
                        migra_bin_reco=np.insert(migra_bin_reco, 0, (migra_mean[i_min + 1] - migra_e_reco[i]))
                        migra_bin_reco=np.append(migra_bin_reco, migra_e_reco[i + 1] - migra_mean[i_max])
                    val=self.evaluate(offset=offset, e_true=e_true, migra=migra_mean[index])
                    integral[i]=np.sum(val * migra_bin_reco) / norm
        return integral

    def plot_migration(self, ax=None, offset=None, e_true=None,
                       migra=None, **kwargs):
        """Plot energy dispersion for given offset and true energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`, optional
            Offset
        e_true : `~gammapy.utils.energy.Energy`, optional
            True energy
        migra : `~numpy.array`, list, optional
            Migration nodes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax=plt.gca() if ax is None else ax

        if offset is None:
            offset=Angle([1], 'deg')
        else:
            offset=np.atleast_1d(Angle(offset))
        if e_true is None:
            e_true=Energy([0.1, 1, 10], 'TeV')
        else:
            e_true=np.atleast_1d(Energy(e_true))
        migra=self.migra if migra is None else migra

        for ener in e_true:
            for off in offset:
                disp=self.evaluate(offset=off, e_true=ener, migra=migra)
                label='offset = {0:.1f}\nenergy = {1:.1f}'.format(off, ener)
                ax.plot(migra, disp, label=label, **kwargs)

        ax.set_xlabel('E_Reco / E_True')
        ax.set_ylabel('Probability density')
        ax.legend(loc='upper left')

        return ax

    def plot_bias(self, ax=None, offset=None, e_true=None,
                  migra=None, **kwargs):
        """Plot migration as a function of true energy for a given offset

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`, optional
            Offset
        e_true : `~gammapy.utils.energy.Energy`, optional
            True energy
        migra : `~numpy.array`, list, optional
            Migration nodes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        from matplotlib.colors import PowerNorm
        import matplotlib.pyplot as plt

        kwargs.setdefault('cmap', 'afmhot')
        kwargs.setdefault('norm', PowerNorm(gamma=0.5))

        ax=plt.gca() if ax is None else ax

        if offset is None:
            offset=Angle([1], 'deg')
        if e_true is None:
            e_true=self.energy
        if migra is None:
            migra=self.migra

        z=self.evaluate(offset=offset, e_true=e_true, migra=migra)
        x=e_true.value
        y=migra

        ax.pcolor(x, y, z, **kwargs)
        ax.semilogx()
        ax.set_xlabel('Energy (TeV)')
        ax.set_ylabel('E_Reco / E_true')

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : (float, float)
            Size of the resulting plot
        """
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.plot_bias(ax=axes[0])
        self.plot_migration(ax=axes[1])
        edisp=self.to_energy_dispersion(offset='1 deg')
        edisp.plot_matrix(ax=axes[2])

        plt.tight_layout()
        plt.show()
        return fig

    def _prepare_linear_interpolator(self, interp_kwargs):
        from scipy.interpolate import RegularGridInterpolator

        x=self.offset
        y=self.migra
        z=np.log10(self.energy.value)
        points=(x, y, z)
        values=self.dispersion

        self._linear=RegularGridInterpolator(points, values, **interp_kwargs)

    def info(self):
        """Print some basic info.
        """
        ss="\nSummary EnergyDispersion2D info\n"
        ss += "--------------------------------\n"
        # Summarise data members
        ss += array_stats_str(self.energy, 'energy')
        ss += array_stats_str(self.offset, 'offset')
        ss += array_stats_str(self.migra, 'migra')
        ss += array_stats_str(self.dispersion, 'dispersion')

        energy=Energy('1 TeV')
        e_reco=EnergyBounds([0.8, 1.2], 'TeV')
        offset=Angle('0.5 deg')
        p=self.get_response(offset, energy, e_reco)[0]

        ss += 'Probability to reconstruct a {} photon in the range {} at {}' \
              ' offset: {:.2f}'.format(energy, e_reco, offset, p)

        return ss
