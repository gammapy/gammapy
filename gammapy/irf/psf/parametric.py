# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import logging
import numpy as np
from astropy import units as u
from gammapy.maps import MapAxis, MapAxes
from gammapy.utils.gauss import MultiGauss2D
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from .core import PSF


__all__ = ["ParametricPSF", "EnergyDependentMultiGaussPSF", "PSFKing"]

log = logging.getLogger(__name__)


class ParametricPSF(PSF):
    """Parametric PSF base class

    Parameters
    -----------
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
        pass

    @abc.abstractmethod
    def evaluate(self, rad):
        pass

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

    @unit.setter
    def unit(self, values):
        self._unit = {key: u.Unit(val) for key, val in values.items()}

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
        table = self.axes.to_table(format="gadf-dl3")

        for name in self.required_parameters:
            table[name.upper()] = self.data[name].T[np.newaxis]
            table[name.upper()].unit = self.unit[name]

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
        axes = MapAxes.from_table(table, format=format)[cls.required_axes]

        dtype = {
            "names": cls.required_parameters,
            "formats": len(cls.required_parameters) * (np.float32,)
        }

        data = np.empty(axes.shape, dtype=dtype)
        unit = {}

        for name in cls.required_parameters:
            column = table[name.upper()]
            values = column.data[0].transpose()

            # TODO: this fixes some files where sigma is written as zero
            if "SIGMA" in name:
                values[values == 0] = 1.

            data[name] = values.reshape(axes.shape)
            unit[name] = column.unit or ""

        return cls(
            axes=axes,
            data=data,
            meta=table.meta.copy(),
            unit=unit
        )

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
        from gammapy.irf import PSF3D
        from gammapy.datasets.map import RAD_AXIS_DEFAULT

        offset_axis = self.axes["offset"]
        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad_axis = RAD_AXIS_DEFAULT.center
        else:
            rad_axis = MapAxis.from_edges(rad, name="rad")

        axes = MapAxes([energy_axis_true, offset_axis, rad_axis])
        data = self.evaluate(**axes.get_coord())

        return PSF3D(
            axes=axes,
            data=data.value,
            unit=data.unit,
            meta=self.meta.copy()
        )

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n\n"
        str_ += f"\taxes      : {self.axes.names}\n"
        str_ += f"\tshape     : {self.data.shape}\n"
        str_ += f"\tndim      : {len(self.axes)}\n"
        str_ += f"\tparameters: {self.required_parameters}\n"
        return str_.expandtabs(tabsize=2)


class EnergyDependentMultiGaussPSF(ParametricPSF):
    """Triple Gauss analytical PSF depending on energy and theta.

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
    Plot R68 of the PSF vs. theta and energy:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EnergyDependentMultiGaussPSF
        filename = '$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits'
        psf = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
        psf.plot_containment(0.68)
        plt.show()
    """
    tag = "psf_3gauss"
    required_axes = ["energy_true", "offset"]
    required_parameters = ["SIGMA_1", "SIGMA_2", "SIGMA_3", "SCALE", "AMPL_2", "AMPL_3"]

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
        m = MultiGauss2D(**pars)
        m.normalize()
        containment = m.containment_fraction(rad)
        return containment

    def evaluate_parameters(self, energy_true, offset):
        """"""
        energy = u.Quantity(energy_true)
        offset = u.Quantity(offset)

        sigmas, norms = [], []

        pars = {"A_1": 1}

        for name in ["SIGMA_1", "SIGMA_2", "SIGMA_3"]:
            sigma = self._interpolators[name]((energy, offset))
            sigmas.append(sigma)

        for name, interp_name in zip(["scale", "A_2", "A_3"], ["SCALE", "AMPL_2", "AMPL_3"]):
            interp = self._interpolators[interp_name]
            pars[name] = interp((energy, offset))

        for idx, sigma in enumerate(sigmas):
            a = pars[f"A_{idx + 1}"]
            norm = (pars["scale"] * 2 * a * sigma ** 2).to_value(sigma.unit ** 2)
            norms.append(norm)

        return {"norms": norms, "sigmas": sigmas}

    def evaluate(self, rad, energy_true, offset):
        """Evaluate psf model"""
        pars = self.evaluate_parameters(energy_true=energy_true, offset=offset)

        m = MultiGauss2D(**pars)
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
    default_interp_kwargs = dict(
        bounds_error=False, fill_value=None
    )

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
        sigma, gamma = pars["sigma"], pars["gamma"]

        term_1 = -(1 + rad ** 2 / (2 * gamma * sigma ** 2)) ** -gamma
        term_2 = rad ** 2 + 2 * gamma * sigma ** 2
        term_3 = 2 * gamma * sigma ** 2

        with np.errstate(divide="ignore", invalid="ignore"):
            containment = term_1 * term_2 / term_3

        return containment

    def evaluate(self, rad, energy_true, offset):
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
        pars = self.evaluate_parameters(
            energy_true=energy_true, offset=offset
        )
        sigma, gamma = pars["sigma"], pars["gamma"]

        with np.errstate(divide="ignore"):
            term1 = 1 / (2 * np.pi * sigma ** 2)
            term2 = 1 - 1 / gamma
            term3 = (1 + rad ** 2 / (2 * gamma * sigma ** 2)) ** (-gamma)

        return term1 * term2 * term3

