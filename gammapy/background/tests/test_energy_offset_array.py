from ..energy_offset_array import EnergyOffsetArray


def test_energy_offset_array():
    a = EnergyOffsetArray(1, 2, 3)


    assert a.energy == 1
    assert a.offset == 2
    assert a.data == 3
