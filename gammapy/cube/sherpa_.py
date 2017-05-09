# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..utils.energy import EnergyBounds
from sherpa.astro.ui import erf
import astropy.wcs as WCS
from sherpa.models import ArithmeticModel, Parameter, modelCacher1d
from sherpa.data import DataND, BaseData
from sherpa.utils.err import DataErr, NotImplementedErr
from sherpa.utils import SherpaFloat, NoNewAttributesAfterInit, \
    print_fields, create_expr, calc_total_error, bool_cast, \
    filter_bins, interpolate, linear_interp

"""
Definition of the model NormGauss2DInt: Integrated 2D gaussian
"""
fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
fwhm_to_sigma_erf = np.sqrt(2) * fwhm_to_sigma


class NormGauss2DInt(ArithmeticModel):
    """Integrated 2D gaussian for sherpa models
    """

    def __init__(self, name='normgauss2dint'):
        # Gauss source parameters
        self.wcs = WCS.WCS()
        self.coordsys = "galactic"  # default
        self.binsize = 1.0
        self.xpos = Parameter(name, 'xpos', 0)  # p[0]
        self.ypos = Parameter(name, 'ypos', 0)  # p[1]
        self.ampl = Parameter(name, 'ampl', 1)  # p[2]
        self.fwhm = Parameter(name, 'fwhm', 1, min=0)  # p[3]
        self.shape = None
        self.n_ebins = None
        ArithmeticModel.__init__(self, name, (self.xpos, self.ypos, self.ampl, self.fwhm))

    def calc(self, p, xlo, xhi, ylo, yhi, *args, **kwargs):
        """
        The normgauss2dint model uses the error function to evaluate the
        the gaussian. This corresponds to an integration over bins.
        """
        return self.normgauss2d(p, xlo, xhi, ylo, yhi)

    def normgauss2d(self, p, xlo, xhi, ylo, yhi):
        sigma_erf = p[3] * fwhm_to_sigma_erf
        return p[2] / 4. * ((erf.calc.calc([1, p[0], sigma_erf], xhi)
                             - erf.calc.calc([1, p[0], sigma_erf], xlo))
                            * (erf.calc.calc([1, p[1], sigma_erf], yhi)
                               - erf.calc.calc([1, p[1], sigma_erf], ylo)))


# This class was copy pasted from sherpa.data.Data2D and modified to account
# for the third dimension
# it is set up to integrate over the energy (first) axis, but not all class
# methods are adapted to that yet (TODO)

class Data3D(DataND):
    """Sherpa 3-D data set.
    """

    def _set_mask(self, val):
        DataND._set_mask(self, val)
        try:
            self._lo = self.apply_filter(self.xlo)
            self._hi = self.apply_filter(self.xhi)
            self._x1 = self.apply_filter(self.x1)
            self._x2 = self.apply_filter(self.x2)
        except DataErr:
            self._hi = self.xhi
            self._lo = self.xlo
            self._x1 = self.x1
            self._x2 = self.x2

    mask = property(DataND._get_mask, _set_mask,
                    doc='Mask array for dependent variable')

    def __init__(self, name, xlo, xhi, x1, x2, y, shape=None, staterror=None,
                 syserror=None):
        self._lo = xlo
        self._hi = xhi
        self._x1 = x1
        self._x2 = x2
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter = bool_cast(filter)
        if filter:
            return (self._lo, self._hi, self._x1, self._x2)
        return (self.xlo, self.xhi, self.x1, self.x2)

    def get_x0(self, filter=False):
        return self.get_indep(filter)[0]

    def get_x1(self, filter=False):
        return self.get_indep(filter)[1]

    def get_x2(self, filter=False):
        return self.get_indep(filter)[2]

    def get_axes(self):
        self._check_shape()
        # FIXME: how to filter an axis when self.mask is size of self.y?
        return (np.arange(self.shape[1]) + 1, np.arange(self.shape[0]) + 1, np.arange(self.shape[0]) + 1)

    def get_dims(self, filter=False):
        # self._check_shape()
        if self.shape is not None:
            return self.shape[::-1]
        return (len(self.get_x0(filter)), len(self.get_x1(filter)), len(self.get_x2(filter)))

    def get_filter_expr(self):
        return ''

    get_filter = get_filter_expr

    def _check_shape(self):
        if self.shape is None:
            raise DataErr('shape', self.name)

    def get_max_pos(self, dep=None):
        if dep is None:
            dep = self.get_dep(True)
        x0 = self.get_x0(True)
        x1 = self.get_x1(True)
        x2 = self.get_x2(True)

        pos = np.asarray(np.where(dep == dep.max())).squeeze()
        if pos.ndim == 0:
            pos = int(pos)
            return (x0[pos], x1[pos], x2[pos])

        return [(x0[index], x1[index], x2[index]) for index in pos]

    def get_img(self, yfunc=None):
        self._check_shape()
        y_img = self.get_y(False, yfunc)
        if yfunc is not None:
            y_img = (y_img[0].reshape(*self.shape),
                     y_img[1].reshape(*self.shape))
        else:
            y_img = y_img.reshape(*self.shape)
        return y_img

    def get_imgerr(self):
        self._check_shape()
        err = self.get_error()
        if err is not None:
            err = err.reshape(*self.shape)
        return err

    def notice(self, x0lo=None, x0hi=None, x1lo=None, x1hi=None, x2lo=None, x2hi=None, ignore=False):
        BaseData.notice(self, (x0lo, x1lo, x2lo), (x0hi, x1hi, x2hi), self.get_indep(),
                        ignore)


class Data3DInt(DataND):
    "3-D integrated data set"

    def _set_mask(self, val):
        DataND._set_mask(self, val)
        try:
            self._x0lo = self.apply_filter(self.x0lo)
            self._x0hi = self.apply_filter(self.x0hi)
            self._x1lo = self.apply_filter(self.x1lo)
            self._x1hi = self.apply_filter(self.x1hi)
            self._x2lo = self.apply_filter(self.x2lo)
            self._x2hi = self.apply_filter(self.x2hi)
        except DataErr:
            self._x0lo = self.x0lo
            self._x1lo = self.x1lo
            self._x0hi = self.x0hi
            self._x1hi = self.x1hi
            self._x2hi = self.x2hi
            self._x2hi = self.x2hi

    mask = property(DataND._get_mask, _set_mask,
                    doc='Mask array for dependent variable')

    def __init__(self, name, x0lo, x1lo, x2lo, x0hi, x1hi, x2hi, y, shape=None,
                 staterror=None, syserror=None):
        self._x0lo = x0lo
        self._x1lo = x1lo
        self._x2lo = x2lo
        self._x0hi = x0hi
        self._x1hi = x1hi
        self._x2hi = x2hi
        BaseData.__init__(self)

    def get_indep(self, filter=False):
        filter = bool_cast(filter)
        if filter:
            return (self._x0lo, self._x1lo, self._x2lo, self._x0hi, self._x1hi, self._x2hi)
        return (self.x0lo, self.x1lo, self.x2lo, self.x0hi, self.x1hi, self.x2hi)

    def get_x0(self, filter=False):
        indep = self.get_indep(filter)
        return (indep[0] + indep[3]) / 2.0

    def get_x1(self, filter=False):
        indep = self.get_indep(filter)
        return (indep[1] + indep[4]) / 2.0

    def get_x2(self, filter=False):
        indep = self.get_indep(filter)
        return (indep[2] + indep[5]) / 2.0

    def notice(self, x0lo=None, x0hi=None, x1lo=None, x1hi=None, x2lo=None, x2hi=None, ignore=False):
        BaseData.notice(self, (x0lo, x1lo, x2hi),
                        (x0hi, x1hi, x2hi), self.get_indep(), ignore)


class CombinedModel3D(ArithmeticModel):
    """
    Combined spatial and spectral 3D model.
    If you ask for a selected region, it will only compare the data and the Combined model on the selected region
    """

    def __init__(self, name='cube-model', spatial_model=None, spectral_model=None):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

        # Fix spectral ampl parameter
        spectral_model.ampl = 1
        spectral_model.ampl.freeze()

        pars = []
        for _ in spatial_model.pars + spectral_model.pars:
            setattr(self, _.name, _)
            pars.append(_)

        self._spatial_pars = slice(0, len(spatial_model.pars))
        self._spectral_pars = slice(len(spatial_model.pars), len(pars))
        ArithmeticModel.__init__(self, name, pars)

    def calc(self, pars, elo, ehi, x, y):
        _spatial = self.spatial_model.calc(pars[self._spatial_pars], x, y)
        _spectral = self.spectral_model.calc(pars[self._spectral_pars], elo, ehi)
        return _spatial * _spectral


class CombinedModel3DInt(ArithmeticModel):
    """
    Combined spatial and spectral 3D model with the possibility to convolve the spatial model*exposure by the PSF.
    If you ask for a selected region, it will only compare the data and the Combined model on the selected region

    Parameters
    ----------
    coord: `~astropy.coordinates.SkyCoord`
        Position of the edges of the pixel on the sky.
    energies: `~astropy.units.Quantity`
        Reconstructed energy used for the counts cube
    use_psf: bool
        if true will convolve the spatial model by the psf
    exposure: `~gammapy.cube.SkyCube`
        Exposure cube
    psf: `~gammapy.cube.SkyCube`
        Psf cube
    select_region: True
        If True select only the points of the region of interest for the fit
    index_selected_region: tuple
        tuple of three `~numpy.ndarray` containing the indexes of the points of the Cube to keep in the fit (Energy, x, y)
    """

    def __init__(self, coord, energies, name='cube-model', use_psf=True, exposure=None, psf=None, spatial_model=None,
                 spectral_model=None, select_region=False, index_selected_region=None):
        from scipy import signal
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model
        self.use_psf = use_psf
        self.exposure = exposure
        self.psf = psf
        self._fftconvolve = signal.fftconvolve
        xx = coord.data.lon.degree
        yy = coord.data.lat.degree
        self.xx_lo = xx[0:-1, 1:]
        self.xx_hi = xx[0:-1, 0:-1]
        self.yy_lo = yy[0:-1, 0:-1]
        self.yy_hi = yy[1:, 0:-1]
        self.ee_lo = energies[:-1]
        self.ee_hi = energies[1:]
        self.select_region = select_region
        self.index_selected_region = index_selected_region

        # Fix spectral ampl parameter
        spectral_model.ampl = 1
        spectral_model.ampl.freeze()

        pars = []
        for _ in spatial_model.pars + spectral_model.pars:
            setattr(self, _.name, _)
            pars.append(_)

        self._spatial_pars = slice(0, len(spatial_model.pars))
        self._spectral_pars = slice(len(spatial_model.pars), len(pars))
        ArithmeticModel.__init__(self, name, pars)

    def calc(self, pars, elo, xlo, ylo, ehi, xhi, yhi):

        if self.use_psf:
            shape = (len(self.ee_lo), len(self.xx_lo[:, 0]), len(self.xx_lo[0, :]))
            result_convol = np.zeros(shape)
            a = self.spatial_model.calc(pars[self._spatial_pars], self.xx_lo.ravel(), self.xx_hi.ravel(),
                                        self.yy_lo.ravel(), self.yy_hi.ravel()).reshape(self.xx_lo.shape)
            # Convolve the spatial model * exposure by the psf
            for ind_E in range(shape[0]):
                result_convol[ind_E, :, :] = self._fftconvolve(a * self.exposure.data[ind_E, :, :],
                                                               self.psf.data[ind_E, :, :] /
                                                               (self.psf.data[ind_E, :, :].sum()), mode='same')

            spectral_1d = self.spectral_model.calc(pars[self._spectral_pars], self.ee_lo, self.ee_hi)
            if not self.select_region:
                _spatial = result_convol.ravel()
                _spectral = (spectral_1d.reshape(len(self.ee_lo), 1, 1) * np.ones_like(self.xx_lo)).ravel()
            else:
                _spatial = result_convol[self.index_selected_region].ravel()
                _spectral = (spectral_1d.reshape(len(self.ee_lo), 1, 1) * np.ones_like(self.xx_lo))[
                    self.index_selected_region].ravel()

        else:
            _spatial = self.spatial_model.calc(pars[self._spatial_pars], xlo, xhi, ylo, yhi)
            _spectral = self.spectral_model.calc(pars[self._spectral_pars], elo, ehi)
        return _spatial * _spectral


class CombinedModel3DIntConvolveEdisp(ArithmeticModel):
    """
    Combined spatial and spectral 3D model taking into account the energy resolution
     with the possibility to convolve the spatial model*exposure by the PSF.

    Parameters
    ----------
    coord: `~astropy.coordinates.SkyCoord`
        Position of the edges of the pixel on the sky.
    energies: `~astropy.units.Quantity`
        Reconstructed energy used for the counts cube
    use_psf: bool
        if true will convolve the spatial model by the psf
    exposure: `~gammapy.cube.SkyCube`
        Exposure Cube
    psf: `~gammapy.cube.SkyCube`
        Psf cube
    spatial_model: `~sherpa.models`
        spatial sherpa model
    spectral_model: `~sherpa.models`
        spectral sherpa model
    edisp: `~numpy.array`
        2D array in (Ereco,Etrue) for the energy dispersion
    select_region: True
        If True select only the points of the region of interest for the fit
    index_selected_region: tuple
        tuple of three `~numpy.ndarray` containing the indexes of the points of the Cube to keep in the fit (Energy, x, y)

    """

    def __init__(self, coord, energies, name='cube-model', use_psf=True, exposure=None, psf=None, spatial_model=None,
                 spectral_model=None, edisp=None, select_region=False, index_selected_region=None):
        from scipy import signal
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model
        xx = coord.data.lon.degree
        yy = coord.data.lat.degree
        self.xx_lo = xx[0:-1, 1:]
        self.xx_hi = xx[0:-1, 0:-1]
        self.yy_lo = yy[0:-1, 0:-1]
        self.yy_hi = yy[1:, 0:-1]
        self.ee_lo = energies[:-1]
        self.ee_hi = energies[1:]
        self.use_psf = use_psf
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp
        self.true_energy = EnergyBounds(self.exposure.energies("edges"))
        self.dim_x, self.dim_y, self.dim_Ereco, self.dim_Etrue = len(self.xx_lo[:, 0]), len(self.xx_lo[0, :]), \
                                                                 len(self.ee_lo), len(self.true_energy) - 1
        self._fftconvolve = signal.fftconvolve
        # The shape of the counts cube in (Ereco,x,y)
        self.shape_data = (self.dim_Ereco, self.dim_x, self.dim_y)
        # Array that will store the result after multipliying by the energy resolution in (x,y,Etrue,Ereco)
        self.convolve_edisp = np.zeros(
            (self.dim_x, self.dim_y, self.dim_Etrue, self.dim_Ereco))
        self.select_region = select_region
        self.index_selected_region = index_selected_region

        # Fix spectral ampl parameter
        spectral_model.ampl = 1
        spectral_model.ampl.freeze()

        pars = []
        for _ in spatial_model.pars + spectral_model.pars:
            setattr(self, _.name, _)
            pars.append(_)

        self._spatial_pars = slice(0, len(spatial_model.pars))
        self._spectral_pars = slice(len(spatial_model.pars), len(pars))
        ArithmeticModel.__init__(self, name, pars)

    def calc(self, pars, elo, xlo, ylo, ehi, xhi, yhi):
        etrue_centers = self.true_energy.log_centers
        if self.use_psf:
            # Convolve the spatial model * exposure by the psf in etrue
            spatial = np.zeros((self.dim_Etrue, self.dim_x, self.dim_y))
            a = self.spatial_model.calc(pars[self._spatial_pars], self.xx_lo.ravel(), self.xx_hi.ravel(),
                                        self.yy_lo.ravel(), self.yy_hi.ravel()).reshape(self.xx_lo.shape)
            for ind_E in range(self.dim_Etrue):
                spatial[ind_E, :, :] = self._fftconvolve(a * self.exposure.data[ind_E, :, :],
                                                         self.psf.data[ind_E, :, :] /
                                                         (self.psf.data[ind_E, :, :].sum()), mode='same')
                # To avoid nan value for the true energy values asked by the user for which the PSF is not defined.
                # The interpolation gives nan when you are outside the range and when you sum over all the true energy bin to calculate the expected
                # number of counts in the reconstucted energy bin, you get nan whereas you just want the bin in true energy
                # for which the PSF is not defined to not count in the sum.
                spatial[np.isnan(spatial)] = 0
        else:
            spatial_2d = self.spatial_model.calc(pars[self._spatial_pars], self.xx_lo.ravel(), self.xx_hi.ravel(),
                                                 self.yy_lo.ravel(), self.yy_hi.ravel()).reshape(self.xx_lo.shape)
            spatial = np.tile(spatial_2d, (len(etrue_centers), 1, 1))
        # Calculate the spectral model in etrue
        spectral_1d = self.spectral_model.calc(pars[self._spectral_pars], etrue_centers)
        spectral = spectral_1d.reshape(len(etrue_centers), 1, 1) * np.ones_like(self.xx_lo)

        # Convolve by the energy resolution
        etrue_band = self.true_energy.bands
        for ireco in range(self.dim_Ereco):
            self.convolve_edisp[:, :, :, ireco] = (np.rollaxis(spatial, 0, spatial.ndim)
                                                   * np.rollaxis(spectral, 0, spectral.ndim)
                                                   * self.edisp[:, ireco] * etrue_band)
        # Integration in etrue
        sum_model = np.sum(self.convolve_edisp, axis=2)
        model = np.rollaxis(sum_model, -1, 0)
        if not self.select_region:
            return model.ravel()
        else:
            return model[self.index_selected_region].ravel()
