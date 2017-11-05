# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import logging
import numpy as np
import astropy.units as u
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
    mask_dilation_radius : `~astropy.units.Quantity`
        Radius by which mask is dilated with each iteration.
    delete_intermediate_results : bool
        Specify whether results of intermediate iterations should be deleted.
        (Otherwise, these are held in memory). Default True.
    save_intermediate_results : bool
        Specify whether to save intermediate results as FITS files to disk.
        Default False.
    base_dir : str (optional)
        Base of filenames if save_intermediate_results = True. Default 'temp'.

    See Also
    --------
    gammapy.background.RingBackgroundEstimator,
    gammapy.background.AdaptiveRingBackgroundEstimator
    """

    def __init__(self, kernel_src, kernel_bkg,
                 significance_threshold=5, mask_dilation_radius=0.02 * u.deg,
                 delete_intermediate_results=False,
                 save_intermediate_results=False, base_dir='temp'):

        self.parameters = OrderedDict(significance_threshold=significance_threshold,
                                      mask_dilation_radius=mask_dilation_radius,
                                      save_intermediate_results=save_intermediate_results,
                                      delete_intermediate_results=delete_intermediate_results,
                                      base_dir=base_dir)

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
            result_previous = self.images_stack.pop()
            result = self._run_iteration(result_previous)

            if p['delete_intermediate_results']:
                self.images_stack = [result]
            else:
                self.images_stack += [result_previous, result]

            if p['save_intermediate_results']:
                result.write(p['base_dir'] + 'ibe_iteration_{}.fits')

            if self._is_converged(result, result_previous) and (idx >= niter_min):
                log.info('Exclusion mask succesfully converged,'
                         ' after {} iterations.'.format(idx))
                break

        return result

    def _is_converged(self, result, result_previous):
        """Check convergence.

        Criterion: exclusion masks unchanged in subsequent iterations.
        """
        from scipy.ndimage.morphology import binary_fill_holes
        mask = result['exclusion'].data == result_previous['exclusion'].data

        # Because of pixel to pixel noise, the masks can still differ.
        # This is handled by removing structures of the scale of one pixel
        mask = binary_fill_holes(mask)
        return np.all(mask)

    # TODO: make more flexible, e.g. allow using adaptive ring etc.
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

    # TODO: make more flexible, e.g. allow using TS images tec.
    def _estimate_significance(self, counts, background):
        kernel = CustomKernel(self.kernel_src)
        images_lima = compute_lima_image(counts, background, kernel=kernel)
        return images_lima['significance']

    def _run_iteration(self, images):
        """Run one iteration.

        Parameters
        ----------
        images : `gammapy.image.SkyImageList`
            Input sky images
        """
        from scipy.ndimage import binary_erosion
        images.check_required(['counts', 'exclusion', 'background'])
        wcs = images['counts'].wcs.copy()
        p = self.parameters

        significance = self._estimate_significance(images['counts'], images['background'])

        # update exclusion mask
        radius = p['mask_dilation_radius'].to('deg')
        scale = images['counts'].wcs_pixel_scale()[0]
        structure = np.array(Tophat2DKernel((radius / scale).value))

        mask = (significance.data < p['significance_threshold']) | np.isnan(significance)
        mask = binary_erosion(mask, structure, border_value=1)
        exclusion = SkyImage(name='exclusion', data=mask.astype('float'), wcs=wcs)

        background = self._estimate_background(images['counts'], exclusion)
        return SkyImageList([images['counts'], background, exclusion, significance])

    def images_stack_show(self, dpi=120):
        """Show image stack.

        Parameters
        ----------
        dpi : int
            Dots per inch to scale the image.
        """
        import matplotlib.pyplot as plt
        niter_max = len(self.images_stack)
        wcs = self.images_stack[0]['background'].wcs

        height_pix, width_pix = self.images_stack[0]['background'].data.shape
        width = 2 * (width_pix / dpi + 1.)
        height = niter_max * (height_pix / dpi + .5)
        fig = plt.figure(figsize=(width, height))

        for idx, images in enumerate(self.images_stack):
            ax_bkg = fig.add_subplot(niter_max + 1, 2, 2 * idx + 1, projection=wcs)
            bkg = images['background']
            bkg.plot(ax=ax_bkg, vmin=0)
            ax_bkg.set_title('Background, N_iter = {}'.format(idx),
                             fontsize='small')

            ax_sig = fig.add_subplot(niter_max + 1, 2, 2 * idx + 2, projection=wcs)
            sig = images['significance']
            sig.plot(ax=ax_sig, vmin=0, vmax=20)
            ax_sig.set_title('Significance, N_Iter = {}'.format(idx),
                             fontsize='small')
            mask = images['exclusion'].data
            ax_sig.contour(mask, levels=[0], linewidths=2, colors='green')
            if idx < (niter_max - 1):
                for ax in [ax_sig, ax_bkg]:
                    ax.set_xlabel('')
                    ax.coords['glon'].ticklabels.set_visible(False)
            ax_bkg.set_ylabel('')
            ax_sig.set_ylabel('')

        plt.tight_layout(pad=1.08, h_pad=1.5, w_pad=0.2, rect=[0, 0, 1, 0.98])
