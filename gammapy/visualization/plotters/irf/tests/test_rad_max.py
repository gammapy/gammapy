import pytest
import numpy as np
from astropy import units as u
from gammapy.irf import RadMax2D
from gammapy.visualization.plotters.irf import RadMaxPlotter
from gammapy.maps import MapAxis

def test_rad_max_plotter():
    energy_axis = MapAxis.from_energy_bounds(1, 10, nbin=2, unit="TeV", name="energy")
    offset_axis = MapAxis.from_bounds(0, 1, nbin=2, unit="deg", name="offset")
    
    data = np.ones((2, 2)) * 0.1 * u.deg

    rad_max = RadMax2D(
        axes=[energy_axis, offset_axis],
        data=data,
        unit="deg",
    )
    
    plotter = RadMaxPlotter()
    ax = plotter.plot(rad_max=rad_max)
    
    assert ax is not None