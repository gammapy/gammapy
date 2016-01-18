from ..energy_offset_array import EnergyOffsetArray
from ...datasets import gammapy_extra

@requires_data('gammapy-extra')
def test_energy_offset_array():
    a = EnergyOffsetArray(1, 2, 3)


    assert a.energy == 1
    assert a.offset == 2.4
    assert a.data == 3
