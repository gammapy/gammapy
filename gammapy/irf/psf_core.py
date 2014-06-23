# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility funtions for reading / writing PSF parameters to JSON files.

From the documentaion of load_psf():

The PSF model is convolved with the source model when
set_psf is issued,and the PSF-convolved model is then fit
to data set  when the fit command is run.

Note: it is important that you set xpos and ypos
of your PSF such that it is fully contained in the
current dataset.
Otherwise your PSF will be cut off and your results nonsense!
You can simply set the PSF in the middle of your image
by issueing center_psf()
"""
from __future__ import print_function, division
import json
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy import log
from ..utils.const import sigma_to_fwhm, fwhm_to_sigma
from ..morphology import read_json
from ..morphology import Gauss2DPDF, MultiGauss2D

__all__ = ['GaussPSF',
           'HESSMultiGaussPSF',
           'SherpaMultiGaussPSF',
           'PositionDependentMultiGaussPSF',
           'multi_gauss_psf_kernel'
           ]


class GaussPSF(Gauss2DPDF):
    """Extension of Gauss2D PDF by PSF-specific functionality."""

    def to_hess(self):
        return {'A_1': self.norm, 'sigma_1': self.sigma}

    def to_sherpa(self, binsz):
        """Generate gauss2d parameters for Sherpa such
        that the integral is 1"""
        d = {}
        d['ampl'] = binsz ** 2 * self.norm
        d['fwhm'] = sigma_to_fwhm / binsz * self.sigma
        return {'psf1': d}


class SherpaMultiGaussPSF(object):
    """Multi-Gauss PSF as represented in the Sherpa software.

    Note that Sherpa uses the following function
    f(x,y) = f(r) = A exp[-f(r/F)^2]
    f = 2.7725887 = 4log2 relates the full-width
    at half-maximum F to the Gaussian sigma
    """
    def __init__(self, source):
        if isinstance(source, dict):
            # Assume source is a dict with correct format
            self.pars = source
        # elif isinstance(source, HESS):
            # Get pars dict by from HESS object
            # self.pars = source.to_sherpa()
        elif isinstance(source, str):
            # Assume it is a JSON filename
            fh = open(source)
            self.pars = json.load(fh)
            fh.close()
        else:
            raise ValueError('Unknown source: {0}'.format(source))

    def __str__(self):
        return json.dumps(self.pars, sort_keys=True, indent=4)

    def center_psf(self):
        """Set ``xpos`` and ``ypos`` of the PSF to the dataspace center."""
        import sherpa.astro.ui as sau
        try:
            ny, nx = sau.get_data().shape
            for par in sau.get_psf().kernel.pars:
                if par.name is 'xpos':
                    par.val = (nx + 1) / 2.
                elif par.name is 'ypos':
                    par.val = (ny + 1) / 2.
        except:
            raise Exception('PSF is not centered.')

    def set(self):
        """Set the PSF for Sherpa."""
        import sherpa.astro.ui as sau
        # from morphology.utils import read_json
        read_json(self.pars, sau.set_model)
        sau.load_psf('psf', sau.get_model())
        sau.set_psf('psf')
        self.center_psf()

    def eval(self, t, ampl1, fwhm1, ampl2, fwhm2, ampl3, fwhm3):
        """Hand-coded eval for debugging."""
        f = 4 * np.log(2)
        psf1 = ampl1 * np.exp(-f * t ** 2 / fwhm1 ** 2)
        psf2 = ampl2 * np.exp(-f * t ** 2 / fwhm2 ** 2)
        psf3 = ampl3 * np.exp(-f * t ** 2 / fwhm3 ** 2)
        return psf1 + psf2 + psf3

    def containment_fraction(self, theta, npix=1000):
        """Compute fraction of PSF contained inside theta."""
        import sherpa.astro.ui as sau
        sau.dataspace2d((npix, npix))
        self.set()
        # x_center = get_psf().kernel.pars.xpos
        # y_center = get_psf().kernel.pars.ypos
        x_center, y_center = sau.get_psf().model.center
        x_center, y_center = x_center + 0.5, y_center + 0.5  # shift seen on image.
        x, y = sau.get_data().x0, sau.get_data().x1
        # @note Here we have to use the source image, before I used
        # get_model_image(), which returns the PSF-convolved PSF image,
        # which is a factor of sqrt(2) ~ 1.4 too wide!!!
        p = sau.get_source_image().y.flatten()
        p /= np.nansum(p)
        mask = (x - x_center) ** 2 + (y - y_center) ** 2 < theta ** 2
        fraction = np.nansum(p[mask])
        if 0:  # debug
            sau.get_data().y = p
            sau.save_data('psf_sherpa.fits', clobber=True)
            sau.get_data().y = mask.astype('int')
            sau.save_data('mask_sherpa.fits', clobber=True)
        return fraction


class HESSMultiGaussPSF(object):
    """Multi-Gauss PSF as represented in the HESS software.

    The 2D Gaussian is represented as a 1D exponential
    probability density function per offset angle squared:
    dp / dtheta**2 = [0]*(exp(-x/(2*[1]*[1]))+[2]*exp(-x/(2*[3]*[3]))

    @param source: either a dict of a filename

    The following two parameters control numerical
    precision / speed. Usually the defaults are fine.
    @param theta_max: Maximum offset in numerical computations
    @param npoints: Number of points in numerical computations
    @param eps: Allowed tolerance on normalization of total P to 1
    """
    def __init__(self, source):
        if isinstance(source, dict):
            # Assume source is a dict with correct format
            self.pars = source
        else:
            # Assume source is a filename with correct format
            self.pars = self._read_ascii(source)
        # Scale will be computed from normalization anyways,
        # so any default is fine here
        self.pars['scale'] = self.pars.get('scale', 1)
        # This avoids handling the first PSF as a special case
        self.pars['A_1'] = self.pars.get('A_1', 1)

    def _read_ascii(self, filename):
        """Parse file with parameters."""
        fh = open(filename)  # .readlines()
        pars = {}
        for line in fh:
            try:
                key, value = line.strip().split()[:2]
                if key.startswith('#'):
                    continue
                else:
                    pars[key] = float(value)
            except ValueError:
                pass
        fh.close()
        return pars

    def __str__(self):
        return json.dumps(self.pars, sort_keys=True, indent=4)

    def n_gauss(self):
        """Count number of Gaussians."""
        return len([_ for _ in self.pars.keys() if 'sigma' in _])

    def dpdtheta2(self, theta2):
        """dp / dtheta2 at position theta2 = theta ^ 2."""
        theta2 = np.asarray(theta2, 'f')
        total = np.zeros_like(theta2)
        for ii in range(1, self.n_gauss() + 1):
            A = self.pars['A_{0}'.format(ii)]
            sigma = self.pars['sigma_{0}'.format(ii)]
            total += A * np.exp(-theta2 / (2 * sigma ** 2))
        return self.pars['scale'] * total

    def to_sherpa(self, binsz):
        """Convert parameters to Sherpa format.

        @param binsz: Bin size (deg)
        @return: dict of Sherpa parameters

        @note Sherpa uses the following function
        f(x,y) = f(r) = A exp[-f(r/F)^2]
        f = 2.7725887 = 4log2 relates the full-width
        at half-maximum F to the Gaussian sigma.

        @note: The sigma parameters in this class are in
        deg, but in Sherpa we use pix, so we convert here.

        For further explanations on how to convert 1D to 2D Gaussian,
        see the docstring of GCTAResponse::psf_dummy in GammaLib

        Input is unnormalized anyways, so we don't care about absolute
        PSF normalization here, Sherpa automatically renormalizes. -> Check!"""
        pars = {}
        # scale = self.pars['scale']
        for ii in range(1, self.n_gauss() + 1):
            d = {}
            A = self.pars['A_{0}'.format(ii)]
            sigma = self.pars['sigma_{0}'.format(ii)]
            d['ampl'] = A
            d['fwhm'] = sigma_to_fwhm * sigma / binsz
            name = 'psf{0}'.format(ii)
            pars[name] = d
        return pars

    def to_file(self, filename, binsz, fmt='json'):
        """Convert parameters to Sherpa format and write them to a JSON file.
        """
        if fmt == 'json':
            pars = self.to_sherpa(binsz)
            json.dump(pars, open(filename, 'w'),
                      sort_keys=True, indent=4)
        elif fmt == 'ascii':
            fh = open(filename, 'w')
            for name, value in self.pars.items():
                fh.write('{0} {1}\n'.format(name, value))
            fh.close()

    def to_MultiGauss2D(self, normalize=True):
        """Use this to compute containment angles and fractions.

        @note: We have to set norm = 2 * A * sigma ^ 2, because in
        MultiGauss2D norm represents the integral, and in HESS A
        represents the amplitude at 0."""
        sigmas, norms = [], []
        for ii in range(1, self.n_gauss() + 1):
            A = self.pars['A_{0}'.format(ii)]
            sigma = self.pars['sigma_{0}'.format(ii)]
            norm = self.pars['scale'] * 2 * A * sigma ** 2
            sigmas.append(sigma)
            norms.append(norm)
        m = MultiGauss2D(sigmas, norms)
        if normalize:
            m.normalize()
        return m

    def containment_radius(self, containment_fraction):
        """Convolve this PSF with a Gaussian source of width sigma,
        then compute the containment angle of that distribution.
        """
        m = self.to_MultiGauss2D(normalize=True)
        theta = m.containment_radius(containment_fraction)
        return theta


class PositionDependentMultiGaussPSF(object):
    """Position-dependent multi-Gauss PSF.

    Represented by a set of images of multi-Gauss PSF parameters.

    Parameters
    ----------
    hdu_list : `~astropy.io.fits.HDUList`
        HDU list.
    """

    def __init__(self, hdu_list):
        self.hdu_list = hdu_list
        self._psf_pars = ['scale', 'sigma_1', 'A_2', 'sigma_2', 'A_3', 'sigma_3']
        self._psf_vals = dict()
        for psf_par in self._psf_pars:
            self._psf_vals[psf_par] = self.hdu_list[psf_par].data.flat
        self.shape = self.hdu_list['SCALE'].data.shape
        self.size = self.hdu_list['SCALE'].data.size

    @staticmethod
    def read(filename):
        """Read from FITS file.
        """
        hdu_list = fits.open(filename)
        return PositionDependentMultiGaussPSF(hdu_list)

    def containment_radius_image(self, fraction):
        """Compute containment radius image.

        Parameters
        ----------
        fraction : float
            Containment fraction

        Returns
        -------
        image : `numpy.ndarray`
            Containment radius image
        """
        out = np.zeros(self.shape, dtype=float)
        npix = self.size
        for ii in range(npix):
            if (100 * ii) % npix == 0:
                percent = 100. * ii / npix
                log.debug('Processing pixel {ii:5d} of {npix:5d} ({percent:5.2f}%)'
                          ''.format(**locals()))
            try:
                psf = self._get_psf(ii)
                out.flat[ii] = psf.containment_radius(fraction)
            except ValueError:
                # This is what happens to pixels in the map without PSF info
                out.flat[ii] = np.nan

        return out

    def _get_psf(self, index):
        """Get PSF at a given flattened index position.
        """
        psf_vals = [self._psf_vals[parameter][index] for parameter in self._psf_pars]
        psf = HESSMultiGaussPSF(dict(zip(self._psf_pars, psf_vals)))
        return psf


def multi_gauss_psf_kernel(psf_parameters, **kwargs):
    """Create multi-Gauss PSF kernel.

    The Gaussian PSF components are specified via the
    amplitude at the center and the FWHM.
    See the example for the exact format.

    Parameters
    ----------
    psf_parameters : dict
        PSF parameters

    Returns
    -------
    psf_kernel : `astropy.convolution.Kernel`
        PSF kernel

    Examples
    --------
    >>> psf_pars = dict()
    >>> psf_pars['psf1'] = dict(ampl=1, fwhm=2.5)
    >>> psf_pars['psf2'] = dict(ampl=0.06, fwhm=11.14)
    >>> psf_pars['psf3'] = dict(ampl=0.47, fwhm=5.16)
    >>> psf_kernel = multi_gauss_psf_kernel(psf_pars, x_size=51)
    """
    psf = None
    for ii in range(1, 4):
        # Convert sigma and amplitude
        pars = psf_parameters['psf{0}'.format(ii)]
        sigma = fwhm_to_sigma * pars['fwhm']
        ampl = 2 * np.pi * sigma ** 2 * pars['ampl']
        if psf == None:
            psf = float(ampl) * Gaussian2DKernel(sigma, **kwargs)
        else:
            psf += float(ampl) * Gaussian2DKernel(sigma, **kwargs)

    psf.normalize()

    return psf
