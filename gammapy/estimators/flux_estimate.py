# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Table
from gammapy.maps import MapAxis

__all__ = ["FluxEstimate"]


SED_TYPES = ["dnde", "e2dnde", "flux", "eflux"]

OPTIONAL_QUANTITIES = ["err", "errn", "errp", "ul", "scan"]


class FluxEstimate:
    """A flux estimate produced by an Estimator.

    Follows the likelihood SED type description and allows norm values
    to be converted to dnde, flux, eflux and e2dnde

    The flux is converted according to the input spectral model. The latter must be the one used to
    obtain the 'norm' values of the input data.

    The energy axis is obtained from the input data:
    - directly from the energy `MapAxis if the input data is a `dict` of `Map``
    - from the 'e_min' and 'e_max' columns in the input data is an `~astropy.table.Table`

    Parameters
    ----------
    data : dict of `Map` or `Table`
        Mappable containing the sed data with at least a 'norm' entry.
        If data is a Table, it should contain 'e_min' and 'e_max' columns.
    spectral_model : `SpectralModel`
        Reference spectral model used to produce the input data.
    """

    def __init__(self, data, spectral_model):
        # TODO: Check data
        self.data = data

        if hasattr(self.data["norm"], "geom"):
            self.energy_axis = self.data["norm"].geom.axes["energy"]
            self._keys = self.data.keys()
            self._expand_slice = (slice(None), np.newaxis, np.newaxis)
        else:
            # Here we assume there is only one row per energy
            e_edges = self.data["e_min"].quantity
            e_edges = e_edges.insert(len(self.data), self.data["e_max"].quantity[-1])
            self.energy_axis = MapAxis.from_energy_edges(e_edges)
            self._keys = self.data.columns
            self._expand_slice = slice(None)

        # Note that here we could use the specification from dnde_ref to build piecewise PL
        # But does it work beyond min and max centers?
        self.spectral_model = spectral_model

        self._available_quantities = []

        for quantity in OPTIONAL_QUANTITIES:
            norm_quantity = f"norm_{quantity}"
            if norm_quantity in self._keys:
                self._available_quantities.append(quantity)

    # TODO: add support for scan

    def _check_norm_quantity(self, quantity):
        if not quantity in self._available_quantities:
            raise KeyError(
                f"Cannot compute required flux quantity. {quantity} is not defined on current flux estimate."
            )

    @property
    def norm(self):
        return self.data["norm"]

    @property
    def norm_err(self):
        self._check_norm_quantity("err")
        return self.data["norm_err"]

    @property
    def norm_errn(self):
        self._check_norm_quantity("errn")
        return self.data["norm_errn"]

    @property
    def norm_errp(self):
        self._check_norm_quantity("errp")
        return self.data["norm_errp"]

    @property
    def norm_ul(self):
        self._check_norm_quantity("ul")
        return self.data["norm_ul"]

    @property
    def dnde_ref(self):
        result = self.spectral_model(self.energy_axis.center)
        return result[self._expand_slice]

    @property
    def e2dnde_ref(self):
        result = (
            self.spectral_model(self.energy_axis.center) * self.energy_axis.center ** 2
        )
        return result[self._expand_slice]

    @property
    def flux_ref(self):
        energy_min = self.energy_axis.edges[:-1]
        energy_max = self.energy_axis.edges[1:]
        result = self.spectral_model.integral(energy_min, energy_max)
        return result[self._expand_slice]

    @property
    def eflux_ref(self):
        energy_min = self.energy_axis.edges[:-1]
        energy_max = self.energy_axis.edges[1:]
        result = self.spectral_model.energy_flux(energy_min, energy_max)
        return result[self._expand_slice]

    @property
    def dnde(self):
        """Return differential flux (dnde) SED values."""
        return self.norm * self.dnde_ref

    @property
    def dnde_err(self):
        """Return differential flux (dnde) SED errors."""
        return self.norm_err * self.dnde_ref

    @property
    def dnde_errn(self):
        """Return differential flux (dnde) SED negative errors."""
        return self.norm_errn * self.dnde_ref

    @property
    def dnde_errp(self):
        """Return differential flux (dnde) SED positive errors."""
        return self.norm_errp * self.dnde_ref

    @property
    def dnde_ul(self):
        """Return differential flux (dnde) SED upper limit."""
        return self.norm_ul * self.dnde_ref

    @property
    def e2dnde(self):
        """Return differential energy flux (e2dnde) SED values."""
        return self.norm * self.e2dnde_ref

    @property
    def e2dnde_err(self):
        """Return differential energy flux (e2dnde) SED errors."""
        return self.norm_err * self.e2dnde_ref

    @property
    def e2dnde_errn(self):
        """Return differential energy flux (e2dnde) SED negative errors."""
        return self.norm_errn * self.e2dnde_ref

    @property
    def e2dnde_errp(self):
        """Return differential energy flux (e2dnde) SED positive errors."""
        return self.norm_errp * self.e2dnde_ref

    @property
    def e2dnde_ul(self):
        """Return differential energy flux (e2dnde) SED upper limit."""
        return self.norm_ul * self.e2dnde_ref

    @property
    def flux(self):
        """Return integral flux (flux) SED values."""
        return self.norm * self.flux_ref

    @property
    def flux_err(self):
        """Return integral flux (flux) SED values."""
        return self.norm_err * self.flux_ref

    @property
    def flux_errn(self):
        """Return integral flux (flux) SED negative errors."""
        return self.norm_errn * self.flux_ref

    @property
    def flux_errp(self):
        """Return integral flux (flux) SED positive errors."""
        return self.norm_errp * self.flux_ref

    @property
    def flux_ul(self):
        """Return integral flux (flux) SED upper limits."""
        return self.norm_ul * self.flux_ref

    @property
    def eflux(self):
        """Return energy flux (eflux) SED values."""
        return self.norm * self.eflux_ref

    @property
    def eflux_err(self):
        """Return energy flux (eflux) SED errors."""
        return self.norm_err * self.eflux_ref

    @property
    def eflux_errn(self):
        """Return energy flux (eflux) SED negative errors."""
        return self.norm_errn * self.eflux_ref

    @property
    def eflux_errp(self):
        """Return energy flux (eflux) SED positive errors."""
        return self.norm_errp * self.eflux_ref

    @property
    def eflux_ul(self):
        """Return energy flux (eflux) SED upper limits."""
        return self.norm_ul * self.eflux_ref
