# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from gammapy.maps import Geom, MapAxis
from gammapy.modeling.models import PowerLawSpectralModel

__all__ = ["FluxEstimate"]


SED_TYPES = ["dnde", "e2dnde", "flux", "eflux"]

OPTIONAL_QUANTITIES = ["err", "errn", "errp", "ul", "scan"]

class FluxEstimate:
    """A flux estimate produced by an Estimator.

    Follows the likelihood SED type description and allows norm values
    to be converted to dnde, flux, eflux and e2dnde

    Parameters
    ----------
    data : dict of `Map` or `Table`
        Mappable containing the sed likelihood data
    spectral_model : `SpectralModel`
        Reference spectral model.
    """
    def __init__(self, data, spectral_model):
        # TODO: Check data
        self.data = data

        if hasattr(self.data["norm"], "Geom"):
            self.energy_axis = self.data["norm"].geom.axes["energy"]
        else:
            # Here we assume there is only one row per energy
            e_edges = self.data["e_min"].quantity
            e_edges = e_edges.insert(len(self.data), self.data["e_max"].quantity[-1])
            self.energy_axis = MapAxis.from_energy_edges(e_edges)

        # Note that here we could use the specification from dnde_ref to build piecewise PL
        # But does it work beyond min and max centers?
        self.spectral_model = spectral_model

        for quantity in OPTIONAL_QUANTITIES:
            name_property = f"norm_{quantity}"
            setattr(self, name_property, self.data[name_property])

    @property
    def dnde_ref(self):
        return self.spectral_model(self.energy_axis.center)

    @property
    def e2dnde_ref(self):
        return self.spectral_model(self.energy_axis.center)*self.energy_axis.center**2

    @property
    def flux_ref(self):
        energy_min = self.energy_axis.edges[:-1]
        energy_max = self.energy_axis.edges[1:]
        return self.spectral_model.integral(energy_min, energy_max)

    @property
    def eflux_ref(self):
        energy_min = self.energy_axis.edges[:-1]
        energy_max = self.energy_axis.edges[1:]
        return self.spectral_model.energy_flux(energy_min, energy_max)

    @property
    def norm(self):
        return self.data["norm"]
"""
    @property
    def norm_err(self):
        return self.data["norm_err"]

    @property
    def norm_errn(self):
        return self.data["norm_errn"]

    @property
    def norm_errp(self):
        return self.data["norm_errp"]

    @property
    def norm_ul(self):
        return self.data["norm_ul"]

    @property
    def dnde(self):
        return self.dnde_ref * self.norm

    @property
    def dnde_err(self):
        return self.dnde_ref * self.norm_err

    @property
    def dnde_errn(self):
        return self.dnde_ref * self.norm_errn

    @property
    def dnde_errp(self):
        return self.dnde_ref * self.norm_errp

    @property
    def dnde_ul(self):
        return self.dnde_ref * self.norm_ul

    @property
    def e2dnde(self):
        return self.e2dnde_ref * self.norm

    @property
    def e2dnde_err(self):
        return self.e2dnde_ref * self.norm_err

    @property
    def e2dnde_errn(self):
        return self.e2dnde_ref * self.norm_errn

    @property
    def e2dnde_errp(self):
        return self.e2dnde_ref * self.norm_errp

    @property
    def e2dnde_ul(self):
        return self.e2dnde_ref * self.norm_ul

    @property
    def flux(self):
        return self.flux_ref * self.norm

    @property
    def flux_err(self):
        return self.flux_ref * self.norm_err

    @property
    def flux_errn(self):
        return self.flux_ref * self.norm_errn

    @property
    def flux_errp(self):
        return self.flux_ref * self.norm_errp

    @property
    def eflux_ul(self):
        return self.eflux_ref * self.norm_ul

    @property
    def eflux(self):
        return self.eflux_ref * self.norm

    @property
    def eflux_err(self):
        return self.eflux_ref * self.norm_err

    @property
    def eflux_errn(self):
        return self.eflux_ref * self.norm_errn

    @property
    def eflux_errp(self):
        return self.eflux_ref * self.norm_errp

    @property
    def eflux_ul(self):
        return self.eflux_ref * self.norm_ul

"""