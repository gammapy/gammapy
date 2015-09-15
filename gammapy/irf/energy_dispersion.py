# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.units import Quantity
from ..spectrum.energy import EnergyBounds

__all__ = [
    'EnergyDispersion',
    'gauss_energy_dispersion_matrix',
    'EnergyDispersion2D',
]


class EnergyDispersion(object):

    """Energy dispersion matrix.

    Parameters
    ----------
    pdf_matrix : array_like
        2-dim energy dispersion matrix (probability density).
        First index for true energy, second index for reco energy.
    e_true : `~gammapy.spectrum.EnergyBounds`
        True energy binning
    e_reco : `~gammapy.spectrum.EnergyBounds`
        Reco energy binning

    Notes
    -----
    We use a dense matrix (`numpy.ndarray`) for the energy dispersion matrix.
    An alternative would be to store a sparse matrix (`scipy.sparse.csc_matrix`).
    It's not clear which would be more efficient for typical gamma-ray
    energy dispersion matrices.

    The most common file format for energy dispersion matrices is the
    RMF (Redistribution Matrix File) format from X-ray astronomy:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
    """
    DEFAULT_PDF_THRESHOLD = 1e-6

    def __init__(self, pdf_matrix, e_true, e_reco=None,
                 pdf_threshold=DEFAULT_PDF_THRESHOLD):
        
        if not isinstance(e_true, EnergyBounds) or not isinstance(
            e_reco, EnergyBounds):
            raise ValueError("Energies must be Energy objects")

        self._pdf_matrix = np.asarray(pdf_matrix)
        self._e_true = e_true
        if e_reco is None:
            self._e_reco = np.asarray(e_true)
        else:
            self._e_reco = e_reco

        self._pdf_threshold = pdf_threshold
        self._interpolate2d_func = None

    @property
    def pdf_threshold(self):
        """PDF matrix zero-suppression threshold (float)"""
        return self._pdf_threshold

    @pdf_threshold.setter
    def pdf_threshold(self, value):
        if self._pdf_threshold > value:
            ss = 'Lowering the PDF matrix zero-suppression threshold can lead to incorrect results.\n'
            ss += 'Old PDF threshold: {0}\n'.format(self._pdf_threshold)
            ss += 'New PDF threshold: {0}'.format(value)
            raise Exception(ss)

        self._pdf_threshold = value

        m = self._pdf_matrix
        m[m < value] = 0

    @property
    def reco_energy(self):
        """Reconstructed Energy axis (`~gammapy.spectrum.EnergyBounds`)
        """
        return self._e_reco

    def true_energy(self):
        """Reconstructed Energy axis (`~gammapy.spectrum.EnergyBounds`)
        """
        return self._e_true

    def __str__(self):
        ss = 'Energy dispersion information:\n'
        ss += 'PDF matrix threshold: {0}\n'.format(self.pdf_threshold)
        m = self._pdf_matrix
        ss += 'PDF matrix filling factor: {0}\n'.format(np.sum(m > 0) / m.size)
        ss += 'True energy range: {0}\n'.format(self.energy_range('true'))
        ss += 'Reco energy range: {0}\n'.format(self.energy_range('reco'))
        return ss

    @classmethod
    def read(cls, filename, format='RMF'):
        """Create `EnergyDispersion` from FITS file.

        Parameters
        ----------
        filename : str
            File name
        format : {'RMF'}
            File format
        """
        if format == 'RMF':
            return EnergyDispersion._read_rmf(filename)
        else:
            ss = 'No reader defined for format "{0}".\n'.format(format)
            ss += 'Available formats: RMF'
            raise ValueError(ss)

    @classmethod
    def _read_rmf(cls, filename):
        """Create `EnergyDispersion` object from RMF data.
        """
        hdu_list = fits.open(filename)
        return cls.from_hdu_list(hdu_list)

    @classmethod
    def from_hdu_list(cls, hdu_list):
        """Create `EnergyDispersion` object from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``MATRIX`` and ``EBOUNDS`` extensions.
        """
        data = hdu_list['MATRIX'].data
        header = hdu_list['MATRIX'].header

        pdf_matrix = np.zeros([len(data), header['DETCHANS']], dtype=np.float64)

        for i, l in enumerate(data):
            if l.field('N_GRP'):
                m_start = 0
                for k in range(l.field('N_GRP')):
                    pdf_matrix[i, l.field('F_CHAN')[k]: l.field('F_CHAN')[k] + l.field('N_CHAN')[k]] = l.field('MATRIX')[m_start:m_start + l.field('N_CHAN')[k]]
                    m_start += l.field('N_CHAN')[k]

        pdf_threshold = float(header['LO_THRES'])

        e_reco = EnergyBounds.from_ebounds(hdu_list['EBOUNDS'])
        e_true = EnergyBounds.from_rmf_matrix(hdu_list['MATRIX'])

        return cls(pdf_matrix, e_true, e_reco, pdf_threshold)

    def write(self, filename, *args, **kwargs):
        """Write RMF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(filename, *args, **kwargs)
        
    def to_fits(self, header=None, energy_unit='TeV'):
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

        #Quick hack around np_to_rmf
        #not tested at all
        telescope='DUMMY'
        instrument='DUMMY'
        filter='NONE'
        minprob = 0.001
        rm = self._pdf_matrix
        erange = self._e_true
        ebounds = self._e_reco

        # Intialize the arrays to be used to construct the RM extension
        n_rows = len(rm)
        energy_lo = np.zeros(n_rows)  # Low energy bounds
        energy_hi = np.zeros(n_rows)  # High energy bounds
        n_grp = []  # Number of channel subsets
        f_chan = []  # First channels in each subset
        n_chan = []  # Number of channels in each subset
        matrix = []  # Matrix elements

        # Loop over the matrix and fill the arrays
        for i, r in enumerate(rm):
            energy_lo[i] = erange[i]
            energy_hi[i] = erange[i + 1]
            # Create mask for all matrix row values above the minimal probability
            m = r > minprob
            # Intialize variables & arrays for the row
            n_grp_row, n_chan_row_c = 0, 0
            f_chan_row, n_chan_row, matrix_row = [], [], []
            new_subset = True
            # Loop over row entries and fill arrays appropriately
            for j, v in enumerate(r):
                if m[j]:
                    if new_subset:
                        n_grp_row += 1
                        f_chan_row.append(j)
                        new_subset = False
                    matrix_row.append(v)
                    n_chan_row_c += 1
                else:
                    if not new_subset:
                        n_chan_row.append(n_chan_row_c)
                        n_chan_row_c = 0
                        new_subset = True
        if not new_subset:
            n_chan_row.append(n_chan_row_c)
        n_grp.append(n_grp_row)
        f_chan.append(f_chan_row)
        n_chan.append(n_chan_row)
        matrix.append(matrix_row)

            # Create RMF FITS table extension from data
        tbhdu = fits.new_table(
            [fits.Column(name='ENERG_LO',
                         format='1E',
                         array=energy_lo,
                         unit='TeV'),
             fits.Column(name='ENERG_HI',
                         format='1E',
                         array=energy_hi,
                         unit='TeV'),
             fits.Column(name='N_GRP',
                         format='1I',
                         array=n_grp),
             fits.Column(name='F_CHAN',
                         format='PI()',
                         array=f_chan),
             fits.Column(name='N_CHAN',
                         format='PI()',
                         array=n_chan),
             fits.Column(name='MATRIX',
                         format='PE(()',
                         array=matrix)
             ]
            )

    # Write FITS extension header

        chan_min, chan_max, chan_n = 0, rm.shape[1] - 1, rm.shape[1]

        header = tbhdu.header
        header['EXTNAME'] = 'MATRIX', 'name of this binary table extension'
        header['TLMIN4'] = chan_min, 'First legal channel number'
        header['TLMAX4'] = chan_max, 'Highest legal channel number'
        header['TELESCOP'] = telescope, 'Mission/satellite name'
        header['INSTRUME'] = instrument, 'Instrument/detector'
        header['FILTER'] = filter, 'Filter information'
        header['CHANTYPE'] = 'PHA', 'Type of channels (PHA, PI etc)'
        header['DETCHANS'] = chan_n, 'Total number of detector PHA channels'
        header['LO_THRES'] = minprob, 'Lower probability density threshold for matrix'
        header['HDUCLASS'] = 'OGIP', 'Organisation devising file format'
        header['HDUCLAS1'] = 'RESPONSE', 'File relates to response of instrument'
        header['HDUCLAS2'] = 'RSP_MATRIX', 'Keyword information for Caltools Software.'
        header['HDUVERS '] = '1.3.0', 'Version of file format'
        header['HDUCLAS3'] = 'DETECTOR', 'Keyword information for Caltools Software.'
        header['CCNM0001'] = 'MATRIX', 'Keyword information for Caltools Software.'
        header['CCLS0001'] = 'CPF', 'Keyword information for Caltools Software'
        header['CDTP0001'] = 'DATA', 'Keyword information for Caltools Software.'
        
    # UTC date when this calibration should be first used (yyyy-mm-dd)
        header['CVSD0001'] = '2011-01-01 ', 'Keyword information for Caltools Software.'

    # UTC time on the dat when this calibration should be first used (hh:mm:ss)
        header['CVST0001'] = '00:00:00', 'Keyword information for Caltools Software.'
        
    # String giving a brief summary of this data set
        header['CDES0001'] = 'dummy data', 'Keyword information for Caltools Software.'

    # Optional, but maybe useful (taken from the example in the RMF/ARF document)
        header['CBD10001'] = 'CHAN({0}- {1})'.format(chan_min, chan_max), 'Keyword information for Caltools Software.'
        header['CBD20001'] = 'ENER({0}-{1})TeV'.format(erange[0], erange[-1]), 'Keyword information for Caltools Software.'

    # Obsolet RMF headers, included for the benefit of old software
        header['RMFVERSN'] = '1992a', 'Obsolete'
        header['HDUVERS1'] = '1.1.0', 'Obsolete'
        header['HDUVERS2'] = '1.2.0', 'Obsolete'

    # Create EBOUNDS FITS table extension from data
        tbhdu2 = fits.new_table(
            [fits.Column(name='CHANNEL',
                         format='1I',
                         array=np.arange(len(ebounds) - 1)),
             fits.Column(name='E_MIN',
                         format='1E',
                         array=ebounds[:-1],
                         unit='TeV'),
             fits.Column(name='E_MAX',
                         format='1E',
                         array=ebounds[1:],
                         unit='TeV')
             ]
            )

        chan_min, chan_max, chan_n = 0, rm.shape[0] - 1, rm.shape[0]

        header = tbhdu2.header
        header['EXTNAME'] = 'EBOUNDS', 'Name of this binary table extension'
        header['TELESCOP'] = telescope, 'Mission/satellite name'
        header['INSTRUME'] = instrument, 'Instrument/detector'
        header['FILTER'] = filter, 'Filter information'
        header['CHANTYPE'] = 'PHA', 'Type of channels (PHA, PI etc)'
        header['DETCHANS'] = chan_n, 'Total number of detector PHA channels'
        header['TLMIN1'] = chan_min, 'First legal channel number'
        header['TLMAX1'] = chan_max, 'Highest legal channel number'
        header['HDUCLASS'] = 'OGIP', 'Organisation devising file format'
        header['HDUCLAS1'] = 'RESPONSE', 'File relates to response of instrument'
        header['HDUCLAS2'] = 'EBOUNDS', 'This is an EBOUNDS extension'
        header['HDUVERS'] = '1.2.0', 'Version of file format'
        header['HDUCLAS3'] = 'DETECTOR', 'Keyword information for Caltools Software.'
        header['CCNM0001'] = 'EBOUNDS', 'Keyword information for Caltools Software.'
        header['CCLS0001'] = 'CPF', 'Keyword information for Caltools Software.'
        header['CDTP0001'] = 'DATA', 'Keyword information for Caltools Software.'
        
        # UTC date when this calibration should be first used (yyyy-mm-dd)
        header['CVSD0001'] = '2011-01-01 ', 'Keyword information for Caltools Software.'

        # UTC time on the dat when this calibration should be first used (hh:mm:ss)
        header['CVST0001'] = '00:00:00', 'Keyword information for Caltools Software.'

        # Optional - name of the PHA file for which this file was produced
    #header('PHAFILE', '', 'Keyword information for Caltools Software.')

        # String giving a brief summary of this data set
        header['CDES0001'] = 'dummy description', 'Keyword information for Caltools Software.'

    # Obsolet EBOUNDS headers, included for the benefit of old software
        header['RMFVERSN'] = '1992a', 'Obsolete'
        header['HDUVERS1'] = '1.0.0', 'Obsolete'
        header['HDUVERS2'] = '1.1.0', 'Obsolete'

        # Create primary HDU and HDU list to be stored in the output file
        # TODO: can remove PrimaryHDU here?
        hdu_list = fits.HDUList([fits.PrimaryHDU(), tbhdu, tbhdu2])

        return hdu_list

    def __call__(self, energy_true, energy_reco, method='step'):
        """Compute energy dispersion.

        Available evalutation methods:

        * ``"step"`` -- TODO
        * ``"interpolate2d"`` -- TODO

        Parameters
        ----------
        energy_true : array_like
            True energy
        energy_reco : array_like
            Reconstructed energy
        method : {'interpolate2d'}
            Evaluation method

        Returns
        -------
        pdf : array
            Probability density dP / dlog10E_reco
        """
        x = np.log10(energy_true)
        y = np.log10(energy_reco)
        if method == 'interpolate2d':
            self._interpolate2d(x, y)
        else:
            ss = 'Invalid method: {0}\n'.format(method)
            ss += 'Available methods: matrix, bias'
            raise ValueError(ss)

    def _interpolate2d(self, x, y):
        if self._interpolate2d_func is None:
            # TODO: set up spline representation
            self._interpolate2d_func = 42
        else:
            return self._interpolate2d_func(x, y)

    def apply(self, data):
        """Apply energy dispersion.

        Computes the matrix product of ``data``
        (which typically is model flux or counts in true energy bins)
        with the energy dispersion matrix.

        Parameters
        ----------
        data : array_like
            1-dim data array.

        Returns
        -------
        convolved_data : array
            1-dim data array after multiplication with the energy dispersion matrix
        """
        return np.dot(data, self._pdf_matrix)

    def plot(self, type='matrix', energy=None):
        """Create energy dispersion plot.

        Parameters
        ----------
        type : {'matrix', 'bias', 'pdf_energy_true', 'pdf_energy_reco'}
            Type of plot to generate
        energy : array_like
            Energies at which to plot the PDF.
            Only applies to type 'pdf_energy_true' and 'pdf_energy_reco'.

        Examples
        --------
        TODO
        """
        if type == 'matrix':
            self._plot_matrix()
        elif type == 'bias':
            self._plot_bias()
        elif type == 'pdf_energy_true':
            self._plot_pdf(energy, 'true')
        elif type == 'pdf_energy_reco':
            self._plot_pdf(energy, 'reco')
        else:
            ss = 'Invalid type: {0}\n'.format(type)
            ss += 'Available types: matrix, bias'
            raise ValueError(ss)

    @property
    def _extent(self):
        """Extent (x0, x1, y0, y1) for plotting (4x float)

        x stands for true energy and y for reconstructed energy
        """
        x = self.energy_range('true')
        y = self.energy_range('reco')
        return x[0], x[1], y[0], y[1]

    def _plot_matrix(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        image = self._pdf_matrix
        plt.imshow(image, extent=self._extent, interpolation='none',
                   origin='lower', norm=LogNorm())
        plt.xlabel('True energy (TeV)')
        plt.ylabel('Reco energy (TeV)')
        plt.loglog()

        # TODO: better colorbar formatting
        plt.colorbar()

        plt.tight_layout()

    def _plot_bias(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.xlabel('True energy (TeV)')
        plt.ylabel('Reco energy (TeV)')

    def _plot_pdf(self, energy, axis):
        raise NotImplementedError

def gauss_energy_dispersion_matrix(ebounds, sigma=0.2):
    """Create Gaussian energy dispersion matrix.

    TODO: this is Gaussian in e_reco ... should be log(e_reco) I think.

    TODO: give formula: Gaussian in log(e_reco)

    TODO: add option to add poisson noise

    TODO: extend to have a vector of bias and resolution for various true energies.

    Parameters
    ----------
    ebounds : array_like
        1-dim energy binning array (TeV)
    sigma : float
        RMS width of Gaussian energy dispersion.

    Returns
    -------
    pdf_matrix : array
        PDF matrix
    """
    from scipy.special import erf

    nbins = len(ebounds) - 1
    logerange = np.log10(ebounds)

    logemingrid = logerange[:-1] * np.ones([nbins, nbins])
    logemaxgrid = logerange[1:] * np.ones([nbins, nbins])
    logecentergrid = np.transpose(((logerange[:-1] + logerange[1:]) / 2.) * np.ones([nbins, nbins]))

    # gauss = lambda p, x: p[0] / np.sqrt(2. * np.pi * p[2] ** 2.) * np.exp(- (x - p[1]) ** 2. / 2. / p[2] ** 2.)
    gauss_int = lambda p, x_min, x_max: .5 * (erf((x_max - p[1]) / np.sqrt(2. * p[2] ** 2.)) - erf((x_min - p[1]) / np.sqrt(2. * p[2] ** 2.)))

    pdf_matrix = gauss_int([1., 10. ** logecentergrid, sigma * 10. ** logecentergrid], 10. ** logemingrid, 10. ** logemaxgrid)

    return pdf_matrix

    # hdu_list = np_to_rmf(rm, ea_erange, ea_erange, 1E-5,
    #                     telescope=telescope, instrument=instrument)
    # return hdu_list


class EnergyDispersion2D(object):

    """Offset-dependent energy dispersion matrix.

    Parameters
    ----------
    etrue_lo : `~numpy.ndarray`, list
        True energy lower bounds
    etrue_hi : `~numpy.ndarray`, list
        True energy upper bounds
    migra_lo : `~numpy.ndarray`, list
        Migration lower bounds
    migra_hi : `~numpy.ndarray`, list
        Migration upper bounds
    offset_lo : `~numpy.ndarray`, list
        Offset lower bounds
    offset_hi : `~numpy.ndarray`, list
        Offset lower bounds
    dispersion : `~numpy.ndarray`
        PDF matrix

    Examples
    --------
    
    Plot migration histogram for a given offset and true energy

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDispersion2D
        from gammapy.datasets import get_path
        filename = get_path("../test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz",
        location='remote')
        edisp2D = EnergyDispersion2D.read(filename)
        edisp2D.plot_migration()
        plt.xlim(0, 4)


    Plot evolution of bias and resolution as a function of true energy
    for a given offset

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        import numpy as np
        from gammapy.irf import EnergyDispersion2D
        from gammapy.spectrum.energy import Energy
        from astropy.coordinates import Angle
        from gammapy.datasets import get_path
        filename = get_path("../test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz",
        location='remote')
        edisp2D = EnergyDispersion2D.read(filename)
        migra = np.linspace(0.1,2,80)
        e_true = Energy.equal_log_spacing(0.13,60,60,'TeV')
        offset = Angle([0.554], 'deg')
        edisp2D.plot_bias(offset=offset, e_true=e_true, migra=migra)
        plt.xscale('log')

    Write RMF fits table


    """

    def __init__(self, etrue_lo, etrue_hi, migra_lo, migra_hi, offset_lo,
                 offset_hi, dispersion):

        if not isinstance(etrue_lo, Quantity) or not isinstance(etrue_hi, Quantity):
            raise ValueError("Energies must be Quantity objects.")
        if not isinstance(offset_lo, Angle) or not isinstance(offset_hi, Angle):
            raise ValueError("Offsets must be Angle objects.")

        self.migra_lo = migra_lo
        self.migra_hi = migra_hi
        self.offset_lo = offset_lo
        self.offset_hi = offset_hi
        self.dispersion = dispersion

        self.ebounds = EnergyBounds.from_lower_and_upper_bounds(etrue_lo, etrue_hi)
        self.energy = self.ebounds.log_centers
        self.offset = (offset_hi + offset_lo) / 2
        self.migra = (migra_hi + migra_lo) / 2

        self._prepare_linear_interpolator()

    @classmethod
    def from_fits(cls, hdu):
        """Create `EnergyDispersion2D` from ``GCTAEdisp2D`` format HDU.

        Parameters
        ----------
        hdu : `~astropy.io.fits.BinTableHDU`
            ``ENERGY DISPERSION`` extension.
            
        """

        data = hdu.data
        header = hdu.header
        e_lo = EnergyBounds(data['ETRUE_LO'].squeeze(), header['TUNIT1'])
        e_hi = EnergyBounds(data['ETRUE_HI'].squeeze(), header['TUNIT2'])
        o_lo = Angle(data['THETA_LO'].squeeze(), header['TUNIT5'])
        o_hi = Angle(data['THETA_HI'].squeeze(), header['TUNIT6'])
        m_lo = data['MIGRA_LO'].squeeze()
        m_hi = data['MIGRA_HI'].squeeze()
        matrix = data['MATRIX'].squeeze()

        return cls(e_lo, e_hi, m_lo, m_hi, o_lo, o_hi, matrix)

    @classmethod
    def read(cls, filename):
        """Create `EnergyDispersion2D` from ``GCTAEdisp2D`` format FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        hdulist = fits.open(filename)
        return cls.from_fits(hdulist['ENERGY DISPERSION'])

    def evaluate(self, offset=None, e_true=None, migra=None):
        """Probability for a given offset, true energy, and migration
        
        Parameters
        ----------
        e_true : `~gammapy.spectrum.EnergyBounds`, None
            True energy axis
        migra : `~numpy.ndarray`
            Energy migration e_reco/e_true
        offset : `~astropy.coordinates.Angle`
            Offset
        """

        if offset is None:
            offset = self.offset
        if e_true is None:
            e_true = self.energy
        if migra is None:
            migra = self.migra

        if not isinstance(e_true, Quantity):
            raise ValueError("Energy must be a Quantity object.")
        if not isinstance(offset, Angle):
            raise ValueError("Offset must be an Angle object.")

        offset = offset.to('degree')
        e_true = e_true.to('TeV')

        val = self._eval(offset=offset, e_true=e_true, migra=migra)

        return val

    def _eval(self, offset=None, e_true=None, migra=None):

        x = np.asarray(offset.value)
        y = np.asarray(migra)
        z = np.asarray(np.log10(e_true.value))
        ax = [x, y, z]

        in_shape = (ax[0].size, ax[1].size, ax[2].size)

        for i, s in enumerate(ax):
            if ax[i].shape == ():
                ax[i] = ax[i].reshape(1)

        # TODO: There is a bug here that could be investigated
        # When energy[0] is given to the interpolator is out if bounds
        # This does not happen when an array (e.g. energy[0:2]) is given

        pts = [[xx, yy, zz] for xx in ax[0] for yy in ax[1] for zz in ax[2]]

        val_array = self._linear(pts)
        return val_array.reshape(in_shape).squeeze()

    def to_energy_dispersion(self, e_reco, offset, e_true=None):
        """Detector response R(Delta E_reco, Delta E_true)

        Probability to reconstruct an energy in a given true energy band
        in a given reconstructed energy band

        Parameters
        ----------
        e_reco : `~gammapy.spectrum.EnergyBounds`
            Reconstructed energy axis
        offset : `~astropy.coordinates.Angle`
            Offset
        e_true : `~gammapy.spectrum.EnergyBounds`, None
            True energy axis

        Returns
        -------
        edisp : `~gammapy.irf.EnergyDispersion`
            Energy disperion matrix
        """

        if e_true is None:
            e_true = self.ebounds

        energy = e_true.log_centers

        rm = []

        for ener in energy:
            vec = self.get_response(offset, ener, e_reco=e_reco)
            rm.append(vec)

        return EnergyDispersion(rm, e_true, e_reco)

    def get_response(self, offset, e_true, e_reco=None):
        """Detector response R(Delta E_reco, E_true)

        Probability to reconstruct a given true energy in a given reconstructed
        energy band

        Parameters
        ----------
        e_true : `~gammapy.spectrum.Energy`
            True energy axis
        e_reco : `~gammapy.spectrum.EnergyBounds`, None
            Reconstructed energy axis
        offset : `~astropy.coordinates.Angle`
            Offset

        Returns
        -------
        rv : `~numpy.ndarray`
            Redistribution vector
        """

        if e_reco is None:
            e_reco = EnergyBounds.from_lower_and_upper_bounds(
                self.migra_lo * e_true, self.migra_hi * e_true)
            migra = self.migra

        else:
            center = e_reco.log_centers
            migra = center / e_true

        band = e_reco.bands

        val = self.evaluate(offset=offset, e_true=e_true, migra=migra)
        rv = (val / e_true) * band

        return rv.value

    def plot_migration(self, ax=None, offset=None, energy=None,
                       migra=None, **kwargs):
        """Plot energy dispersion for given offset and true energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            val = self.offset
            offset = Angle([1], 'deg')
        if energy is None:
            energy = Quantity([0.2, 10], 'TeV')
        if migra is None:
            migra = self.migra

        for ener in energy:
            for off in offset:
                disp = self.evaluate(offset=off, e_true=ener, migra=migra)
                label = 'offset = {0:.1f}\nenergy = {1:.1f}'.format(off, ener)
                plt.plot(migra, disp, label=label, **kwargs)

        plt.xlabel('E_Reco / E_True')
        plt.ylabel('Probability')
        plt.legend(loc='upper right')

        return ax

    def plot_bias(self, ax=None, offset=None, e_true=None,
                  migra=None, **kwargs):
        """Plot migration as a function of true energy for a given offset
        """
        
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = Angle([1], 'deg')
        if e_true is None:
            e_true = self.energy
        if migra is None:
            migra = self.migra

        z = self.evaluate(offset=offset, e_true=e_true, migra=migra)
        x = e_true.value
        y = migra

        plt.pcolor(x, y, z, **kwargs)

        return ax

    def _prepare_linear_interpolator(self):
        """Linear interpolation in N dimensions

        Values outside the bounds are set to 0
        """
        from scipy.interpolate import RegularGridInterpolator

        x = self.offset
        y = self.migra
        z = np.log10(self.energy.value)
        data = self.dispersion

        self._linear = RegularGridInterpolator(
            (x, y, z), data, bounds_error=False, fill_value=0)
