"""Plot an energy dispersion using a gaussian parametrisation"""
import matplotlib.pyplot as plt
from gammapy.irf import EDispKernel
from gammapy.maps import MapAxis

energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=10)
energy_axis_true = MapAxis.from_energy_bounds(
    "0.5 TeV", "30 TeV", nbin=10, per_decade=True, name="energy_true"
)


edisp = EDispKernel.from_gauss(
    energy_axis=energy_axis, energy_axis_true=energy_axis_true, sigma=0.1, bias=0
)
edisp.peek()
plt.show()
