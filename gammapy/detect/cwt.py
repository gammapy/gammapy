# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel, MexicanHat2DKernel
from ..maps import WcsNDMap, MapAxis, WcsGeom

__all__ = ["CWT", "CWTData", "CWTKernels"]

log = logging.getLogger(__name__)


def difference_of_gauss_kernel(radius, scale_step, n_sigmas=8):
    """Difference of 2 Gaussians (i.e. Mexican hat) kernel array.

    TODO: replace by http://astropy.readthedocs.io/en/latest/api/astropy.convolution.MexicanHat2DKernel.html
    once there are tests in place that establish the algorithm
    """
    sizex = int(n_sigmas * scale_step * radius)
    sizey = int(n_sigmas * scale_step * radius)
    radius = float(radius)
    xc = 0.5 * sizex
    yc = 0.5 * sizey
    y, x = np.mgrid[0 : sizey - 1, 0 : sizex - 1]
    x = x - xc
    y = y - yc
    x1 = x / radius
    y1 = y / radius
    g1 = np.exp(-0.5 * (x1 ** 2 + y1 ** 2))
    g1 = g1 / (2 * np.pi * radius ** 2)
    x1 = x1 / scale_step
    y1 = y1 / scale_step
    g2 = np.exp(-0.5 * (x1 ** 2 + y1 ** 2))
    g2 = g2 / (2 * np.pi * radius ** 2 * scale_step ** 2)
    return g1 - g2


class CWT(object):
    """Continuous wavelet transform.

    TODO: describe algorithm (modify the words below)

    Depending on their spectral index, sources won't have the same characteristic scale.
    Therefore to detect sources, we need to compute the wavelet transform at several
    scales in order to search for various PSF sizes. Then for each scale, the wavelet
    transform values under the given significance threshold are rejected. This gives us
    a multiscale support. Then, using the reconstruction by continuous wavelet packets,
    we obtain a filtered image yielding the detected sources.
    To compute the threshold image for a given scale a, the standard EGRET diffuse
    background model to which was added the flux of the extragalactic background,
    and the exposure for the considered energy range were used.

    Parameters
    ----------
    kernels : `~gammapy.detect.CWTKernels`
        Kernels for the algorithm.
    max_iter : int, optional (default 10)
        The maximum number of iterations of the CWT algorithm.
    tol : float, optional (default 1e-5)
        Tolerance for stopping criterion.
    significance_threshold : float, optional (default 3.0)
        Measure of statistical significance.
    significance_island_threshold : float, optional (default None)
        Measure is used for cleaning of isolated pixel islands
        that are slightly above ``significance_threshold``.
    remove_isolated : boolean, optional (default True)
        If ``True``, isolated pixels will be removed.
    keep_history : boolean, optional (default False)
        Save cwt data from all the iterations.

    References
    ----------
    R. Terrier et al (2001) "Wavelet analysis of EGRET data"
    See http://adsabs.harvard.edu/abs/2001ICRC....7.2923T
    """

    def __init__(
        self,
        kernels,
        max_iter=10,
        tol=1e-5,
        significance_threshold=3.0,
        significance_island_threshold=None,
        remove_isolated=True,
        keep_history=False,
    ):
        self.kernels = kernels
        self.max_iter = max_iter
        self.tol = tol
        self.significance_threshold = significance_threshold
        self.significance_island_threshold = significance_island_threshold
        self.remove_isolated = remove_isolated
        self.history = [] if keep_history else None

        # previous_variance is initialized on the first iteration
        self.previous_variance = None

    def _execute_iteration(self, data):
        """Do one iteration of the algorithm.

        Parameters
        ----------
        data : `~gammapy.detect.CWTData`
            Images.
        """
        self._transform(data=data)
        self._compute_support(data=data)
        self._inverse_transform(data=data)

    def _transform(self, data):
        """Do the transform itself.

        The transform is made by using `scipy.signal.fftconvolve`.

        TODO: document.

        Parameters
        ----------
        data : `~gammapy.detect.CWTData`
            Images for transform.
        """
        from scipy.signal import fftconvolve

        total_background = data._model + data._background + data._approx
        excess = data._counts - total_background
        log.debug("Excess sum: {0:.4f}".format(excess.sum()))
        log.debug("Excess max: {0:.4f}".format(excess.max()))

        log.debug("Computing transform and error")
        for idx_scale, kern in self.kernels.kern_base.items():
            data._transform_3d[idx_scale] = fftconvolve(excess, kern, mode="same")
            data._error[idx_scale] = np.sqrt(
                fftconvolve(total_background, kern ** 2, mode="same")
            )
        log.debug("Error sum: {0:.4f}".format(data._error.sum()))
        log.debug("Error max: {0:.4f}".format(data._error.max()))

        log.debug("Computing approx and approx_bkg")
        data._approx = fftconvolve(
            data._counts - data._model - data._background,
            self.kernels.kern_approx,
            mode="same",
        )
        data._approx_bkg = fftconvolve(
            data._background, self.kernels.kern_approx, mode="same"
        )
        log.debug("Approximate sum: {0:.4f}".format(data._approx.sum()))
        log.debug("Approximate background sum: {0:.4f}".format(data._approx_bkg.sum()))

    def _compute_support(self, data):
        """Compute the multiresolution support with hard sigma clipping.

        Imposing a minimum significance on a connected region of significant pixels
        (i.e. source detection).

        TODO: document?
        What's happening here:
        - calc significance for all scales
        - for each scale compute support:
            - create mask, whether significance value more than significance_threshold
            - find all separated structures on mask, label them (with help
              of 'scipy.ndimage.label')
            - for each structure do:
                - find pixels coordinates of the structure
                - if just one pixel, we can remove it from mask if we want
                - if the max value of significance in that structure less
                  than significance_island_threshold, remove this structure from mask
            - update support by mask

        Parameters
        ----------
        data : `~gammapy.detect.CWTData`
            Images after transform.
        """
        from scipy.ndimage import label

        log.debug("Computing significance")
        significance = data._transform_3d / data._error

        log.debug("For each scale start to compute support")
        for idx_scale in range(self.kernels.n_scale):
            log.debug(
                "Start to compute support for scale "
                "{:.2f}".format(self.kernels.scales[idx_scale])
            )

            log.debug(
                "Create mask based on significance "
                "threshold {:.2f}".format(self.significance_threshold)
            )
            mask = significance[idx_scale] > self.significance_threshold

            # Produce a list of connex structures in the support
            labeled_mask, n_structures = label(mask)
            for struct_label in range(n_structures):
                coords = np.where(labeled_mask == struct_label + 1)

                # Remove isolated pixels from support
                if self.remove_isolated and coords[0].size == 1:
                    log.debug("Remove isolated pixels from support")
                    mask[coords] = False

                if self.significance_island_threshold is not None:
                    # If maximal significance of the structure does not reach significance
                    # island threshold, remove significant pixels island from support
                    struct_signif = significance[idx_scale][coords]
                    if struct_signif.max() < self.significance_island_threshold:
                        log.debug(
                            "Remove significant pixels island {} from support".format(
                                struct_label + 1
                            )
                        )
                        mask[coords] = False

            log.debug(
                "Update support for scale {:.2f}".format(self.kernels.scales[idx_scale])
            )
            data._support[idx_scale] |= mask

        log.debug("Support sum: {}".format(data._support.sum()))

    def _inverse_transform(self, data):
        """Do the inverse transform (reconstruct the image).

        TODO: describe better what this does.

        Parameters
        ----------
        data : `~gammapy.detect.CWTData`
            Images for inverse transform.
        """
        data._transform_2d = np.sum(data._support * data._transform_3d, axis=0)
        log.debug("Update model")
        data._model += data._transform_2d * (data._transform_2d > 0)
        log.debug("Model sum: {:.4f}".format(data._model.sum()))
        log.debug("Model max: {:.4f}".format(data._model.max()))

    def _is_converged(self, data):
        """Check if the algorithm has converged on current iteration.

        TODO: document metric used, but not super important.

        Parameters
        ----------
        data : `~gammapy.detect.CWTData`
            Images after iteration.

        Returns
        -------
        answer : boolean
            Answer if CWT has converged.
        """
        log.debug("Check the convergence")
        residual = data._counts - (data._model + data._approx)
        variance = residual.var()
        log.info("Residual sum: {0:.4f}".format(residual.sum()))
        log.info("Residual max: {0:.4f}".format(residual.max()))
        log.info("Residual variance: {0:.4f}".format(residual.var()))

        if self.previous_variance is None:
            self.previous_variance = variance
            return False

        variance_ratio = abs(
            (self.previous_variance - variance) / self.previous_variance
        )
        log.info("Variance ratio: {:.7f}".format(variance_ratio))

        self.previous_variance = variance
        return variance_ratio < self.tol

    def analyze(self, data):
        """Run iterative filter peak algorithm.

        The algorithm modifies the original data.

        Parameters
        ----------
        data : `~gammapy.detect.CWTData`
            Input images.
        """
        if self.history is not None:
            self.history = [copy.deepcopy(data)]

        for n_iter in range(self.max_iter):
            log.info("************ Start iteration {} ************".format(n_iter + 1))
            self._execute_iteration(data=data)
            if self.history is not None:
                log.debug("Save current data")
                self.history.append(copy.deepcopy(data))
            converge_answer = self._is_converged(data=data)
            if converge_answer:
                break

        if converge_answer:
            log.info("Convergence reached at iteration {}".format(n_iter + 1))
        else:
            log.info(
                "Convergence not formally reached at iteration {}".format(n_iter + 1)
            )


class CWTKernels(object):
    """Conduct arrays of kernels and scales for CWT algorithm.

    Parameters
    ----------
    n_scale : int
        Number of scales.
    min_scale : float
        First scale used.
    step_scale : float
        Base scaling factor.
    old : boolean (default False)
        DEBUG attribute. If False, use astropy MaxicanHat kernels for kernel_base.

    Attributes
    ----------
    n_scale : int
        Number of scales considered.
    min_scale : float
        First scale used.
    step_scale : float
        Base scaling factor.
    scales : `~numpy.ndarray`
        Grid of scales.
    kern_base : dict
        Dictionary of scale powers as keys and 2D kernel arrays.
        (mexican hat) as values
    kern_approx : `~numpy.ndarray`
        2D Gaussian kernel array from maximum scale.

    Examples
    --------
    >>> from gammapy.detect import CWTKernels
    >>> kernels = CWTKernels(n_scale=3, min_scale=2.0, step_scale=2.6)
    >>> print (kernels.info_table)
                    Name                        Source
    ---------------------------------- ----------------------
                      Number of scales                      3
                         Minimal scale                    2.0
                            Step scale                    2.6
                                Scales [  2.     5.2   13.52]
                  Kernels approx width                    280
                    Kernels approx sum          0.99986288557
                    Kernels approx max       0.00012877518599
      Kernels base width for 2.0 scale                     40
        Kernels base sum for 2.0 scale      0.000305108917065
        Kernels base max for 2.0 scale        0.0315463182128
      Kernels base width for 5.2 scale                    107
        Kernels base sum for 5.2 scale      0.000158044776015
        Kernels base max for 5.2 scale        0.0050152112595
    Kernels base width for 13.52 scale                    280
      Kernels base sum for 13.52 scale      0.000137114430344
      Kernels base max for 13.52 scale      0.000740731187317
    """

    def __init__(self, n_scale, min_scale, step_scale, old=False):
        self.n_scale = n_scale
        self.min_scale = min_scale
        self.step_scale = step_scale
        self.scales = np.array(
            [min_scale * step_scale ** _ for _ in range(n_scale)], dtype=float
        )

        self.kern_base = {}
        for idx_scale, scale in enumerate(self.scales):
            if old:
                self.kern_base[idx_scale] = difference_of_gauss_kernel(
                    scale, step_scale
                )
            else:
                self.kern_base[idx_scale] = MexicanHat2DKernel(scale * step_scale).array

        max_scale = min_scale * step_scale ** n_scale
        self.kern_approx = Gaussian2DKernel(max_scale).array

    def _info(self):
        """Return information about the object as a dict.

        Returns
        -------
        info : dict
            Information about object with str characteristic as keys and
            characteristic results as values.
        """
        info_dict = {}
        info_dict["Number of scales"] = self.n_scale
        info_dict["Minimal scale"] = self.min_scale
        info_dict["Step scale"] = self.step_scale
        info_dict["Scales"] = str(self.scales)
        info_dict["Kernels approx width"] = len(self.kern_approx)
        info_dict["Kernels approx sum"] = self.kern_approx.sum()
        info_dict["Kernels approx max"] = self.kern_approx.max()

        for idx_scale, scale in enumerate(self.scales):
            info_dict["Kernels base width for {} scale".format(scale)] = len(
                self.kern_base[idx_scale]
            )
            info_dict["Kernels base sum for {} scale".format(scale)] = self.kern_base[
                idx_scale
            ].sum()
            info_dict["Kernels base max for {} scale".format(scale)] = self.kern_base[
                idx_scale
            ].max()

        return info_dict

    @property
    def info_table(self):
        """Summary info table about the object.

        Returns
        -------
        table : `~astropy.table.Table`
            Information about the object.
        """
        info_dict = self._info()

        rows = []
        for name in info_dict:
            rows.append({"Name": name, "Source": info_dict[name]})

        return Table(rows=rows, names=["Name", "Source"])


class CWTData(object):
    """Images for CWT algorithm.

    Contains also input counts and background.

    Parameters
    ----------
    counts : `~gammapy.maps.WcsNDMap`
        2D counts image.
    background : `~gammapy.maps.WcsNDMap`
        2D background image.
    n_scale : int
        Number of scales.

    Examples
    --------
    >>> from gammapy.maps import Map
    >>> from gammapy.detect import CWTData
    >>> filename = '$GAMMAPY_DATA/fermi_survey/all.fits.gz'
    >>> image = Map.read(filename, hdu='COUNTS')
    >>> background = Map.read(filename, hdu='BACKGROUND')
    >>> data = CWTData(counts=image, background=background, n_scale=2)
    """

    def __init__(self, counts, background, n_scale):
        self._counts = np.array(counts.data, dtype=float)
        self._background = np.array(background.data, dtype=float)
        self._geom2d = counts.geom.copy()
        scale_axis = MapAxis(np.arange(n_scale + 1))
        self._geom3d = WcsGeom(
            wcs=counts.geom.wcs, npix=counts.geom.npix, axes=[scale_axis]
        )

        shape_2d = self._counts.shape
        self._model = np.zeros(shape_2d)
        self._approx = np.zeros(shape_2d)
        self._approx_bkg = np.zeros(shape_2d)
        self._transform_2d = np.zeros(shape_2d)

        shape_3d = n_scale, shape_2d[0], shape_2d[1]
        self._transform_3d = np.zeros(shape_3d)
        self._error = np.zeros(shape_3d)
        self._support = np.zeros(shape_3d, dtype=bool)

    @property
    def counts(self):
        """2D counts input image (`~gammapy.maps.WcsNDMap`)."""
        return WcsNDMap(self._geom2d, self._counts)

    @property
    def background(self):
        """2D background input image (`~gammapy.maps.WcsNDMap`)."""
        return WcsNDMap(self._geom2d, self._background)

    @property
    def model(self):
        """2D model image (`~gammapy.maps.WcsNDMap`).

        Positive version of transform_2d image.
        Primordial initialized by zero array.
        """
        return WcsNDMap(self._geom2d, self._model)

    @property
    def approx(self):
        """2D approx ??? image (`~gammapy.maps.WcsNDMap`).

        In the course of iterations updated by convolution of
        ``counts - model - background`` with ``kern_approx``
        Primordial initialized by zero array.
        """
        return WcsNDMap(self._geom2d, self._approx)

    @property
    def approx_bkg(self):
        """2D approx bkg image (`~gammapy.maps.WcsNDMap`).

        In the course of iterations updated by convolution of ``background`` with ``kern_approx``.
        Primordial initialized by zero array.
        """
        return WcsNDMap(self._geom2d, self._approx_bkg)

    @property
    def transform_2d(self):
        """2D transform ??? image (`~gammapy.maps.WcsNDMap`).

        Created from transform_3d by summarize values per 0 axes.
        Primordial initialized by zero array.
        """
        return WcsNDMap(self._geom2d, self._transform_2d)

    @property
    def support_2d(self):
        """2D cube exclusion mask (`~gammapy.maps.WcsNDMap`).

        Created from support_3d by OR-operation per 0 axis.
        """
        support_2d = self._support.sum(0) > 0
        return WcsNDMap(self._geom2d, support_2d)

    @property
    def residual(self):
        """2D residual image (`~gammapy.maps.WcsNDMap`).

        Calculate as ``counts - model - approx``.
        """
        residual = self._counts - (self._model + self._approx)
        return WcsNDMap(self._geom2d, residual)

    @property
    def model_plus_approx(self):
        """TODO: document what this is."""
        return WcsNDMap(self._geom2d, self._model + self._approx)

    @property
    def transform_3d(self):
        """3D transform ??? cube (`~gammapy.maps.WcsNDMap`).

        Primordial initialized by zero array. In the course of
        iterations updated by convolution of ``counts - total_background`` with kernel
        for each scale (``total_background = model + background + approx``).
        """
        return WcsNDMap(self._geom3d, self._transform_3d)

    @property
    def error(self):
        """3D error cube (`~gammapy.maps.WcsNDMap`).

        Primordial initialized by zero array.
        In the course of iterations updated by convolution of ``total_background``
        with kernel^2 for each scale.
        """
        return WcsNDMap(self._geom3d, self._error)

    @property
    def support_3d(self):
        """3D support (exclusion) cube (`~gammapy.maps.WcsNDMap`).

        Primordial initialized by zero array.
        """
        return WcsNDMap(self._geom3d, self._support)

    @property
    def max_scale_image(self):
        """Maximum scale image (`~gammapy.maps.WcsNDMap`)."""
        # Previous version:
        # idx_scale_max = np.argmax(self._transform_3d, axis=0)
        # return kernels.scales[idx_scale_max] * (self._support.sum(0) > 0)
        transform_2d_max = np.max(self._transform_3d, axis=0)
        maximal_image = transform_2d_max * self.support_2d.data
        return WcsNDMap(self._geom2d, maximal_image)

    def __sub__(self, other):
        data = CWTData(
            counts=self.counts,
            background=self.background,
            n_scale=len(self._transform_3d),
        )
        data._model = self._model - other._model
        data._approx = self._approx - other._approx
        data._approx_bkg = self._approx_bkg - other._approx_bkg
        data._transform_2d = self._transform_2d - other._transform_2d

        data._transform_3d = self._transform_3d - other._transform_3d
        data._error = self._error - other._error
        data._support = self._support ^ other._support
        return data

    def images(self):
        """All the images in a dict.

        Returns
        -------
        images : dict
            Dictionary with keys {'counts', 'background', 'model', 'approx',
            'approx_bkg', 'transform_2d', 'maximal', 'support_2d'}
            and 2D `~numpy.ndarray` images as values.
        """
        return dict(
            counts=self.counts,
            background=self.background,
            model=self.model,
            approx=self.approx,
            approx_bkg=self.approx_bkg,
            transform_2d=self.transform_2d,
            model_plus_approx=self.model_plus_approx,
            residual=self.residual,
            maximal=self.max_scale_image,
            support_2d=self.support_2d,
        )

    def cubes(self):
        """All the cubes in a dict.

        Returns
        -------
        cubes : dict
            Dictionary with keys {'transform_3d', 'error', 'support_3d'} and 3D
            `~numpy.ndarray` cubes as values.
        """
        return dict(
            transform_3d=self.transform_3d, error=self.error, support=self.support_3d
        )

    @staticmethod
    def _metrics_info(data, name):
        """Compute variance, mean, find max and min values and compute sum for given data.

        Parameters
        ----------
        data : `~numpy.ndarray`
            2D image or 3D cube.
        name : string
            Name of the data.

        Returns
        -------
        info : dict
            The information about the data.
        """
        return {
            "Name": name,
            "Shape": "2D image" if len(data.shape) == 2 else "3D cube",
            "Variance": data.var(),
            "Mean": data.mean(),
            "Max value": data.max(),
            "Min value": data.min(),
            "Sum values": data.sum(),
        }

    def image_info(self, name):
        """Compute image info.

        Compute variance, mean, find max and min values and compute sum for image with given name.
        Return that information about the image.

        Parameters
        ----------
        name : string
            Name of the image. Name can be as one of the follow: {'counts', 'background',
            'model', 'approx', 'approx_bkg', 'transform_2d', 'model_plus_approx', 'residual',
            'maximal', 'support_2d'}

        Returns
        -------
        table : `~astropy.table.Table`
            Information about the object.
        """
        if name not in self.images():
            raise ValueError(
                "Incorrect name of image. It should be one of the following:"
                "{'counts', 'background', 'model', 'approx', 'approx_bkg', "
                "'transform_2d', 'model_plus_approx', 'residual', 'support_2d', "
                "'maximal'}"
            )

        image = self.images()[name]
        info_dict = self._metrics_info(data=image.data, name=name)

        rows = []
        for metric in info_dict:
            rows.append({"Metrics": metric, "Source": info_dict[metric]})

        return Table(rows=rows, names=["Metrics", "Source"])

    def cube_info(self, name, per_scale=False):
        """Compute cube info.

        Compute variance, mean, find max and min values and compute sum for image with given name.
        Return that information about the image.

        Parameters
        ----------
        name : string
            Name of the image. Name can be as one of the follow: {'transform_3d', 'error', 'support'}
        per_scale : boolean, optional (default False)
            If True, return information about the cube per all the scales.

        Returns
        -------
        table : `~astropy.Table`
            Information about the object.
        """
        if name not in self.cubes():
            raise ValueError(
                "Incorrect name of cube. It should be one of the following:"
                "{'transform_3d', 'error', 'support'}"
            )
        cube = self.cubes()[name]

        rows = []
        if per_scale:
            mask = []
            for index in range(len(cube.data)):
                info_dict = self._metrics_info(data=cube.data[index], name=name)
                for metric in info_dict:
                    rows.append(
                        {
                            "Scale power": index + 1,
                            "Metrics": metric,
                            "Source": info_dict[metric],
                        }
                    )

                # For missing values in `Power scale` column
                scale_mask = np.ones(len(info_dict), dtype=bool)
                scale_mask[0] = False
                mask.extend(scale_mask)

            columns = ["Scale power", "Metrics", "Source"]
            table = Table(rows=rows, names=columns, masked=True)
            table["Scale power"].mask = mask
        elif per_scale is False:
            info_dict = self._metrics_info(data=cube.data, name=name)
            for metric in info_dict:
                rows.append({"Metrics": metric, "Source": info_dict[metric]})
            columns = ["Metrics", "Source"]
            table = Table(rows=rows, names=columns)
        else:
            raise ValueError("Incorrect value for per_scale attribute.")

        return table

    @property
    def info_table(self):
        """Information about all the images and cubes.

        Returns
        -------
        table : `~astropy.Table`
            Information about the object.
        """
        rows = []
        for name, image in self.images().items():
            info_dict = self._metrics_info(data=image.data, name=name)
            rows.append(info_dict)
        for name, cube in self.cubes().items():
            info_dict = self._metrics_info(data=cube.data, name=name)
            rows.append(info_dict)
        columns = rows[0].keys()
        return Table(rows=rows, names=columns)

    def write(self, filename, overwrite=False):
        """Save results to FITS file.

        Parameters
        ----------
        filename : str
            Fits file name.
        overwrite : bool, optional (default False)
            If True, overwrite file with name as filename.
        """
        header = self._geom2d.make_header()
        hdu_list = fits.HDUList()
        hdu_list.append(fits.PrimaryHDU())
        hdu_list.append(fits.ImageHDU(data=self._counts, header=header, name="counts"))
        hdu_list.append(
            fits.ImageHDU(data=self._background, header=header, name="background")
        )
        hdu_list.append(fits.ImageHDU(data=self._model, header=header, name="model"))
        hdu_list.append(fits.ImageHDU(data=self._approx, header=header, name="approx"))
        hdu_list.append(
            fits.ImageHDU(data=self._transform_2d, header=header, name="transform_2d")
        )
        hdu_list.append(
            fits.ImageHDU(data=self._approx_bkg, header=header, name="approx_bkg")
        )
        hdu_list.writeto(filename, overwrite=overwrite)
