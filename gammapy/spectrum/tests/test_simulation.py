# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
from ...utils.testing import requires_dependency
from ...irf import EnergyDispersion, EffectiveAreaTable
from .. import (
    SpectrumExtraction,
    SpectrumSimulation,
    models,
)


@requires_dependency('scipy')
def test_simulation():
    e_true = SpectrumExtraction.DEFAULT_TRUE_ENERGY
    e_reco = SpectrumExtraction.DEFAULT_RECO_ENERGY

    edisp = EnergyDispersion.from_gauss(
        e_true=e_true, e_reco=e_reco, sigma=0.2,
    )

    aeff = EffectiveAreaTable.from_parametrization(energy=e_true)

    model = models.PowerLaw(
        index=2.3 * u.Unit(''),
        amplitude=2.5 * 1e-12 * u.Unit('cm-2 s-1 TeV-1'),
        reference=1 * u.TeV
    )

    sim = SpectrumSimulation(aeff=aeff, edisp=edisp, model=model,
                             livetime=4 * u.h)

    obs = sim.simulate_obs(seed=23)
    assert obs.on_vector.total_counts == 156
