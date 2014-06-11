# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.io import fits

__all__ = ['EnergyDispersion', 'np_to_rmf', 'gauss_energy_dispersion_matrix']


class EnergyDispersion(object):
    """Energy dispersion matrix.

    Parameters
    ----------
    pdf_matrix : array_like
        2-dim energy dispersion matrix (probability density).
        First index for true energy, second index for reco energy.
    energy_true_bounds : array_like
        1-dim true energy binning array (TeV)
    energy_reco_bounds : array_like
        1-dim reco energy binning array (TeV)

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

    def __init__(self, pdf_matrix, energy_true_bounds, energy_reco_bounds=None,
                 pdf_threshold=DEFAULT_PDF_THRESHOLD):
        self._pdf_matrix = np.asarray(pdf_matrix)
        self._energy_true_bounds = np.asarray(energy_true_bounds)
        if energy_reco_bounds == None:
            self._energy_reco_bounds = np.asarray(energy_true_bounds)
        else:
            self._energy_reco_bounds = np.asarray(energy_reco_bounds)

        self._pdf_threshold = pdf_threshold
        self._interpolate2d_func = None

    @property
    def pdf_threshold(self):
        """PDF matrix zero-suppression threshold."""
        return self._pdf_threshold

    @pdf_threshold.setter
    def pdf_threshold(self, value):
        if self._pdf_threshold < value:
            ss = 'Lowering the PDF matrix zero-suppression threshold can lead to incorrect results.\n'
            ss += 'Old PDF threshold: {0}'.format(self._pdf_threshold)
            ss += 'New PDF threshold: {0}'.format(value)
            raise Exception(ss)

        self._pdf_threshold = value
        # Apply new threshold to the matrix
        m = self._pdf_matrix
        m[m < value] = 0

    def energy_bounds(self, axis='true'):
        """Energy bounds array.

        Parameters
        ----------
        axis : {'true', 'reco'}
            Which axis?

        Returns
        -------
        energy_bounds : array
            Energy bounds array.
        """
        if axis == 'true':
            return self._energy_true_bounds
        elif axis == 'reco':
            return self._energy_reco_bounds
        else:
            ss = 'Invalid axis: {0}\n'.format(axis)
            ss += 'Valid options: true, reco'
            raise ValueError(ss)

    def energy_range(self, axis='true'):
        """Energy axis range.

        Parameters
        ----------
        axis : {'true', 'reco'}
            Which axis?

        Returns
        -------
        (emin, emax) : tuple of float
            Energy axis range.
        """
        ebounds = self.energy_bounds(axis)
        return ebounds[0], ebounds[-1]

    def __str__(self):
        ss = 'Energy dispersion information:\n'
        ss += 'PDF matrix threshold: {0}\n'.format(self.pdf_threshold)
        m = self._pdf_matrix
        ss += 'PDF matrix filling factor: {0}\n'.format(np.sum(m > 0) / m.size)
        ss += 'True energy range: {0}\n'.format(self.energy_range('true'))
        ss += 'Reco energy range: {0}\n'.format(self.energy_range('reco'))
        return ss

    @staticmethod
    def read(filename, format='RMF'):
        """Read from file.

        Parameters
        ----------
        filename : str
            File name
        format : {'RMF'}
            File format

        Returns
        -------
        energy_dispersion : `EnergyDispersion`
            Energy dispersion
        """
        if format == 'RMF':
            return EnergyDispersion._read_rmf(filename)
        else:
            ss = 'No reader defined for format "{0}".\n'.format(format)
            ss += 'Available formats: RMF'
            raise ValueError(ss)

    @staticmethod
    def _read_rmf(filename):
        """Create EnergyDistribution object from RMF data.
        """
        hdu_list = fits.open(filename)
        return EnergyDispersion.from_hdu_list(hdu_list)

    @staticmethod
    def from_hdu_list(hdu_list):
        """Create EnergyDistribution object from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``MATRIX`` and ``EBOUNDS`` extensions.
        """
        data = hdu_list['MATRIX'].data
        header = hdu_list['MATRIX'].header

        energy_true_bounds = np.hstack([data['ENERG_LO'], data['ENERG_HI'][-1]])

        pdf_matrix = np.zeros([len(data), header['DETCHANS']], dtype=np.float64)

        for i, l in enumerate(data):
            if l.field('N_GRP'):
                m_start = 0
                for k in range(l.field('N_GRP')):
                    pdf_matrix[i, l.field('F_CHAN')[k] : l.field('F_CHAN')[k] + l.field('N_CHAN')[k]] = l.field('MATRIX')[m_start:m_start + l.field('N_CHAN')[k]]
                    m_start += l.field('N_CHAN')[k]

        pdf_threshold = header['LO_THRES']

        # The reco energy bounds are stored in the 'EBOUNDS' table HDU
        ebounds = hdu_list['EBOUNDS'].data
        energy_reco_bounds = np.hstack([ebounds['E_MIN'], ebounds['E_MAX'][-1]])

        return EnergyDispersion(pdf_matrix, energy_true_bounds, energy_reco_bounds, pdf_threshold)

    def write(self, filename, format='RMF'):
        """Write to file.

        Parameters
        ----------
        filename : str
            File name
        format : {'RMF'}
            File format
        pdf_threshold : float
            Zero suppression threshold for energy distribution matrix
        """
        if format == 'RMF':
            self._write_rmf(filename)
        else:
            ss = 'Invalid format: {0}.\n'.format(format)
            ss += 'Available formats: RMF'
            raise ValueError(ss)

    def _write_rmf(self, filename):
        """Write to file in RMF format."""
        pass

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
        if self._interpolate2d_func == None:
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
        """Extent (x0, x1, y0, y1) for plotting"""
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


def np_to_rmf(rm, erange, ebounds, minprob,
              telescope='DUMMY', instrument='DUMMY', filter='NONE'):
    """Converts a 2D numpy array to an RMF FITS file.

    Parameters
    ----------
    rm : 2D float numpy array
       Energy distribution matrix (probability density) E_true vs E_reco.
    erange : 1D float numpy array
       Bin limits E_true [TeV].
    ebounds : 1D float numpy array
       Bin limits E_reco [TeV].
    minprob : float
        Minimal probability density to be stored in the RMF.
    telescope, instrument, filter : string
        Keyword information for the FITS header.

    Returns
    -------
    hdulist : `~astropy.io.fits.HDUList`
        RMF in HDUList format.

    Notes
    -----
    For more info on the RMF FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
    """
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

    # UTC date when this calibration should be first used (yyy-mm-dd)
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

    # UTC date when this calibration should be first used (yyy-mm-dd)
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

    #hdu_list = np_to_rmf(rm, ea_erange, ea_erange, 1E-5,
    #                     telescope=telescope, instrument=instrument)
    #return hdu_list
