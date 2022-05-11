# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import logging
import numpy as np
from astropy import units as u
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.gauss import MultiGauss2D
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from .core import PSF

__all__ = ["ParametricPSF", "EnergyDependentMultiGaussPSF", "PSFKing"]

log = logging.getLogger(__name__)


class ParametricPSF(PSF):
    """Parametric PSF base class

    Parameters
    ----------
    axes : list of `MapAxis` or `MapAxes`
        Axes
    data : dict of `~numpy.ndarray`, or `~numpy.recarray`
        Data
    unit : dict of str or `~astropy.units.Unit`
        Unit
    meta : dict
        Meta data
    """

    @property
    @abc.abstractmethod
    def required_parameters(self):
        return []

    @abc.abstractmethod
    def evaluate_direct(self, rad, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate_containment(self, rad, **kwargs):
        pass

    def normalize(self):
        """Normalize parametric PSF"""
        raise NotImplementedError

    @property
    def quantity(self):
        """Quantity"""
        quantity = {}

        for name in self.required_parameters:
            quantity[name] = self.data[name] * self.unit[name]

        return quantity

    @property
    def unit(self):
        """Map unit (`~astropy.units.Unit`)"""
        return self._unit

    def to_unit(self, unit):
        """Convert IRF to unit."""
        raise NotImplementedError

    @property
    def _interpolators(self):
        interps = {}

        for name in self.required_parameters:
            points = [a.center for a in self.axes]
            points_scale = tuple([a.interp for a in self.axes])
            interps[name] = ScaledRegularGridInterpolator(
                points, values=self.quantity[name], points_scale=points_scale
            )

        return interps

    def evaluate_parameters(self, energy_true, offset):
        """Evaluate analytic PSF parameters at a given energy and offset.

        Uses nearest-neighbor interpolation.

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        pars = {}
        for name in self.required_parameters:
            value = self._interpolators[name]((energy_true, offset))
            pars[name] = value

        return pars

    def to_table(self, format="gadf-dl3"):
        """Convert PSF table data to table.

        Parameters
        ----------
        format : {"gadf-dl3"}
            Format specification


        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        from gammapy.irf.io import IRF_DL3_HDU_SPECIFICATION

        table = self.axes.to_table(format="gadf-dl3")
        spec = IRF_DL3_HDU_SPECIFICATION[self.tag]["column_name"]

        for name in self.required_parameters:
            column_name = spec[name]
            table[column_name] = self.data[name].T[np.newaxis]
            table[column_name].unit = self.unit[name]

        # Create hdu and hdu list
        return table

    @classmethod
    def from_table(cls, table, format="gadf-dl3"):
        """Create parametric psf from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table  info.

        Returns
        -------
        psf : `~ParametricPSF`
            PSF class
        """
        from gammapy.irf.io import IRF_DL3_HDU_SPECIFICATION

        axes = MapAxes.from_table(table, format=format)[cls.required_axes]

        dtype = {
            "names": cls.required_parameters,
            "formats": len(cls.required_parameters) * (np.float32,),
        }

        data = np.empty(axes.shape, dtype=dtype)
        unit = {}

        spec = IRF_DL3_HDU_SPECIFICATION[cls.tag]["column_name"]

        for name in cls.required_parameters:
            column = table[spec[name]]
            values = column.data[0].transpose()

            # TODO: this fixes some files where sigma is written as zero
            if "sigma" in name:
                values[values == 0] = 1.0

            data[name] = values.reshape(axes.shape)
            unit[name] = column.unit or ""

        unit = {key: u.Unit(val) for key, val in unit.items()}
        return cls(axes=axes, data=data, meta=table.meta.copy(), unit=unit)

    def to_psf3d(self, rad=None):
        """Create a PSF3D from a parametric PSF.

        It will be defined on the same energy and offset values than the input psf.

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad values

        Returns
        -------
        psf3d : `~gammapy.irf.PSF3D`
            PSF3D.
        """
        from gammapy.datasets.map import RAD_AXIS_DEFAULT
        from gammapy.irf import PSF3D

        offset_axis = self.axes["offset"]
        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad_axis = RAD_AXIS_DEFAULT
        else:
            rad_axis = MapAxis.from_edges(rad, name="rad")

        axes = MapAxes([energy_axis_true, offset_axis, rad_axis])
        data = self.evaluate(**axes.get_coord())

        return PSF3D(axes=axes, data=data.value, unit=data.unit, meta=self.meta.copy())

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n\n"
        str_ += f"\taxes      : {self.axes.names}\n"
        str_ += f"\tshape     : {self.data.shape}\n"
        str_ += f"\tndim      : {len(self.axes)}\n"
        str_ += f"\tparameters: {self.required_parameters}\n"
        return str_.expandtabs(tabsize=2)

    def containment(self, rad, **kwargs):
        """Containment of the PSF at given axes coordinates

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        **kwargs : dict
            Other coordinates

        Returns
        -------
        containment : `~numpy.ndarray`
            Containment
        """
        pars = self.evaluate_parameters(**kwargs)
        containment = self.evaluate_containment(rad=rad, **pars)
        return containment

    def evaluate(self, rad, **kwargs):
        """Evaluate the PSF model.

        Parameters
        ----------
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid
        **kwargs : dict
            Other coordinates

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        pars = self.evaluate_parameters(**kwargs)
        value = self.evaluate_direct(rad=rad, **pars)
        return value

    def is_allclose(self, other, rtol_axes=1e-3, atol_axes=1e-6, **kwargs):
        """Compare two data IRFs for equivalency

        Parameters
        ----------
        other : `gammapy.irfs.ParametricPSF`
            The PSF to compare against
        rtol_axes : float
            Relative tolerance for the axes comparison.
        atol_axes : float
            Relative tolerance for the axes comparison.
        **kwargs : dict
                keywords passed to `numpy.allclose`

        Returns
        -------
        is_allclose : bool
            Whether the IRF is all close.
        """
        if not isinstance(other, self.__class__):
            return TypeError(f"Cannot compare {type(self)} and {type(other)}")

        data_eq = True

        for key in self.quantity.keys():
            if self.quantity[key].shape != other.quantity[key].shape:
                return False

            data_eq &= np.allclose(self.quantity[key], other.quantity[key], **kwargs)

        axes_eq = self.axes.is_allclose(other.axes, rtol=rtol_axes, atol=atol_axes)
        return axes_eq and data_eq


def get_sigmas_and_norms(**kwargs):
    """Convert scale and amplitude to norms"""
    sigmas = u.Quantity([kwargs[f"sigma_{idx}"] for idx in [1, 2, 3]])

    scale = kwargs["scale"]
    ones = np.ones(scale.shape)
    amplitudes = u.Quantity([ones, kwargs["ampl_2"], kwargs["ampl_3"]])
    norms = 2 * scale * amplitudes * sigmas**2
    return sigmas, norms


class EnergyDependentMultiGaussPSF(ParametricPSF):
    """Triple Gauss analytical PSF depending on true energy and offset.

    Parameters
    ----------
    axes : list of `MapAxis`
        Required axes are ["energy_true", "offset"]
    data : `~numpy.recarray`
        Data array
    meta : dict
        Meta data

    Examples
    --------
    Plot R68 of the PSF vs. offset and true energy:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDependentMultiGaussPSF
        filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
        psf.plot_containment_radius(fraction=0.68)
        plt.show()
    """

    tag = "psf_3gauss"
    required_axes = ["energy_true", "offset"]
    required_parameters = ["sigma_1", "sigma_2", "sigma_3", "scale", "ampl_2", "ampl_3"]

    @staticmethod
    def evaluate_containment(rad, **kwargs):
        """Containment of the PSF at given axes coordinates

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        **kwargs : dict
            Parameters, see `required_parameters`

        Returns
        -------
        containment : `~numpy.ndarray`
            Containment
        """
        sigmas, norms = get_sigmas_and_norms(**kwargs)
        m = MultiGauss2D(sigmas=sigmas, norms=norms)
        m.normalize()
        containment = m.containment_fraction(rad)
        return containment

    @staticmethod
    def evaluate_direct(rad, **kwargs):
        """Evaluate psf model

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        **kwargs : dict
            Parameters, see `required_parameters`

        Returns
        -------
        value : `~numpy.ndarray`
            PSF value
        """
        sigmas, norms = get_sigmas_and_norms(**kwargs)
        m = MultiGauss2D(sigmas=sigmas, norms=norms)
        m.normalize()
        return m(rad)


class PSFKing(ParametricPSF):
    """King profile analytical PSF depending on energy and offset.

    This PSF parametrisation and FITS data format is described here: :ref:`gadf:psf_king`.

    Parameters
    ----------
    axes : list of `MapAxis` or `MapAxes`
        Data axes, required are ["energy_true", "offset"]
    meta : dict
        Meta data

    """

    tag = "psf_king"
    required_axes = ["energy_true", "offset"]
    required_parameters = ["gamma", "sigma"]
    default_interp_kwargs = dict(bounds_error=False, fill_value=None)

    @staticmethod
    def evaluate_containment(rad, gamma, sigma):
        """Containment of the PSF at given axes coordinates

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        gamma : `~astropy.units.Quantity`
            Gamma parameter
        sigma : `~astropy.units.Quantity`
            Sigma parameter

        Returns
        -------
        containment : `~numpy.ndarray`
            Containment
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            powterm = 1 - gamma
            term = (1 + rad**2 / (2 * gamma * sigma**2)) ** powterm
            containment = 1 - term

        return containment

    @staticmethod
    def evaluate_direct(rad, gamma, sigma):
        """Evaluate the PSF model.

        Formula is given here: :ref:`gadf:psf_king`.

        Parameters
        ----------
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid

        Returns
        -------
        psf_value : `~astropy.units.Quantity`
            PSF value
        """
        with np.errstate(divide="ignore"):
            term1 = 1 / (2 * np.pi * sigma**2)
            term2 = 1 - 1 / gamma
            term3 = (1 + rad**2 / (2 * gamma * sigma**2)) ** (-gamma)

        return term1 * term2 * term3
