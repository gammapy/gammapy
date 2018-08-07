# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions for reading / writing PSF parameters to JSON files.

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
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from ..image.models.gauss import Gauss2DPDF, MultiGauss2D

__all__ = [
    'GaussPSF',
    'HESSMultiGaussPSF',
    'multi_gauss_psf_kernel',
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
        d['fwhm'] = gaussian_sigma_to_fwhm / binsz * self.sigma
        return {'psf1': d}


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
            A = self.pars['A_{}'.format(ii)]
            sigma = self.pars['sigma_{}'.format(ii)]
            total += A * np.exp(-theta2 / (2 * sigma ** 2))
        return self.pars['scale'] * total

    def to_sherpa(self, binsz):
        """Convert parameters to Sherpa format.

        @param binsz: Bin size (deg)
        @return: dict of Sherpa parameters

        Note: Sherpa uses the following function
        f(x,y) = f(r) = A exp[-f(r/F)^2]
        f = 2.7725887 = 4log2 relates the full-width
        at half-maximum F to the Gaussian sigma.

        Note: The sigma parameters in this class are in
        deg, but in Sherpa we use pix, so we convert here.

        For further explanations on how to convert 1D to 2D Gaussian,
        see the docstring of GCTAResponse::psf_dummy in GammaLib

        Input is unnormalized anyways, so we don't care about absolute
        PSF normalization here, Sherpa automatically renormalizes. -> Check!"""
        pars = {}
        # scale = self.pars['scale']
        for ii in range(1, self.n_gauss() + 1):
            d = {}
            A = self.pars['A_{}'.format(ii)]
            sigma = self.pars['sigma_{}'.format(ii)]
            d['ampl'] = A
            d['fwhm'] = gaussian_sigma_to_fwhm * sigma / binsz
            name = 'psf{}'.format(ii)
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
                fh.write('{} {}\n'.format(name, value))
            fh.close()

    def to_MultiGauss2D(self, normalize=True):
        """Use this to compute containment angles and fractions.

        Note: We have to set norm = 2 * A * sigma ^ 2, because in
        MultiGauss2D norm represents the integral, and in HESS A
        represents the amplitude at 0."""
        sigmas, norms = [], []
        for ii in range(1, self.n_gauss() + 1):
            A = self.pars['A_{}'.format(ii)]
            sigma = self.pars['sigma_{}'.format(ii)]
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


def multi_gauss_psf_kernel(psf_parameters, BINSZ=0.02, NEW_BINSZ=0.02, **kwargs):
    """Create multi-Gauss PSF kernel.

    The Gaussian PSF components are specified via the
    amplitude at the center and the FWHM.
    See the example for the exact format.

    Parameters
    ----------
    psf_parameters : dict
        PSF parameters
    BINSZ : float (0.02)
        Pixel size used for the given parameters in deg.
    NEW_BINSZ : float (0.02)
        New pixel size in deg. USed to change the resolution of the PSF.

    Returns
    -------
    psf_kernel : `astropy.convolution.Kernel2D`
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
        pars = psf_parameters['psf{}'.format(ii)]
        sigma = gaussian_fwhm_to_sigma * pars['fwhm'] * BINSZ / NEW_BINSZ
        ampl = 2 * np.pi * sigma ** 2 * pars['ampl']
        if psf is None:
            psf = float(ampl) * Gaussian2DKernel(sigma, **kwargs)
        else:
            psf += float(ampl) * Gaussian2DKernel(sigma, **kwargs)
    psf.normalize()
    return psf
