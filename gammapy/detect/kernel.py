# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy.convolution import Tophat2DKernel, CustomKernel
from ..image import SkyImage, SkyImageList
from .lima import compute_lima_image

log = logging.getLogger(__name__)

__all__ = [
    'KernelBackgroundEstimator',
]


class KernelBackgroundEstimator(object):
    """Estimate background and exclusion mask iteratively.

    Starting from an initial background estimate and exclusion mask
    (both provided, optionally) the algorithm works as follows:

    1. Compute significance image
    2. Create exclusion mask by thresholding significance image
    3. Compute improved background estimate based on new exclusion mask

    The steps are executed repeatedly until the exclusion mask does not change anymore.

    For flexibility the algorithm takes arbitrary source and background kernels.

    Parameters
    ----------
    kernel_src : `numpy.ndarray`
        Source kernel as a numpy array.
    kernel_bkg : `numpy.ndarray`
        Background convolution kernel as a numpy array.
    significance_threshold : float
        Significance threshold above which regions are excluded.
    mask_dilation_radius : `~astropy.coordinates.Angle`
        Radius by which mask is dilated with each iteration.
    keep_record : bool
        Keep record of intermediate results while the algorithm runs?
        Default False.

    See Also
    --------
    gammapy.background.RingBackgroundEstimator,
    gammapy.background.AdaptiveRingBackgroundEstimator
    """

    def __init__(self, kernel_src, kernel_bkg,
                 significance_threshold=5, mask_dilation_radius='0.02 deg',
                 keep_record=False):

        self.parameters = {
            'significance_threshold': significance_threshold,
            'mask_dilation_radius': Angle(mask_dilation_radius),
            'keep_record': keep_record,
        }

        self.kernel_src = kernel_src
        self.kernel_bkg = kernel_bkg
        self.images_stack = []

    def run(self, images, niter_min=2, niter_max=10):
        """Run iterations until mask does not change (stopping condition).

        Parameters
        ----------
        images : `~gammapy.image.SkyImageList`
            Input sky images.
        niter_min : int
            Minimum number of iterations, to prevent early termination of the
            algorithm.
        niter_max : int
            Maximum number of iterations after which the algorithm is
            terminated, if the termination condition (no change of mask between
            iterations) is not already satisfied.

        Returns
        -------
        images : `~gammapy.image.SkyImageList`
            List of sky images containing 'background', 'exclusion' mask and
            'significance' images.
        """
        images.check_required(['counts'])
        p = self.parameters

        # initial mask, if not present
        if 'exclusion' not in images.names:
            images['exclusion'] = SkyImage.empty_like(images['counts'], fill=1)

        # initial background estimate, if not present
        if 'background' not in images.names:
            log.info('Estimating initial background.')
            images['background'] = self._estimate_background(images['counts'],
                                                             images['exclusion'])

        images['significance'] = self._estimate_significance(images['counts'],
                                                             images['background'])
        self.images_stack.append(images)

        for idx in range(niter_max):
            result = self._run_iteration(images)

            if p['keep_record']:
                self.images_stack.append(result)

            if self._is_converged(result, images) and (idx >= niter_min):
                log.info('Exclusion mask succesfully converged,'
                         ' after {} iterations.'.format(idx))
                break

        return result

    def _run_iteration(self, images):
        """Run one iteration.

        Parameters
        ----------
        images : `gammapy.image.SkyImageList`
            Input sky images
        """
        images.check_required(['counts', 'exclusion', 'background'])

        significance = self._estimate_significance(images['counts'], images['background'])
        exclusion = self._estimate_exclusion(images['counts'], significance)
        background = self._estimate_background(images['counts'], exclusion)

        return SkyImageList([images['counts'], background, exclusion, significance])

    def _estimate_significance(self, counts, background):
        kernel = CustomKernel(self.kernel_src)
        images_lima = compute_lima_image(counts.to_wcs_nd_map(), background.to_wcs_nd_map(), kernel=kernel)
        return SkyImage.from_wcs_nd_map(images_lima['significance'])

    def _estimate_exclusion(self, counts, significance):
        from scipy.ndimage import binary_erosion
        wcs = counts.wcs.copy()
        p = self.parameters
        radius = p['mask_dilation_radius'].to('deg').value
        scale = counts.wcs_pixel_scale()[0].value
        mask_dilation_radius_pix = radius / scale
        structure = np.array(Tophat2DKernel(mask_dilation_radius_pix))

        mask = (significance.data < p['significance_threshold']) | np.isnan(significance)
        mask = binary_erosion(mask, structure, border_value=1)
        return SkyImage(name='exclusion', data=mask.astype('float'), wcs=wcs)

    def _estimate_background(self, counts, exclusion):
        """
        Estimate background by convolving the excluded counts image with
        the background kernel and renormalizing the image.
        """
        wcs = counts.wcs.copy()

        # recompute background estimate
        counts_excluded = SkyImage(data=counts.data * exclusion.data, wcs=wcs)
        data = counts_excluded.convolve(self.kernel_bkg, mode='constant')
        norm = exclusion.convolve(self.kernel_bkg, mode='constant')
        return SkyImage(name='background', data=data.data / norm.data, wcs=wcs)

    @staticmethod
    def _is_converged(result, result_previous):
        """Check convergence.

        Criterion: exclusion masks unchanged in subsequent iterations.
        """
        from scipy.ndimage.morphology import binary_fill_holes
        mask = result['exclusion'].data == result_previous['exclusion'].data

        # Because of pixel to pixel noise, the masks can still differ.
        # This is handled by removing structures of the scale of one pixel
        mask = binary_fill_holes(mask)
        return np.all(mask)
