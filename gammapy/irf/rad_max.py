from .core import IRF


class RadMax2D(IRF):
    """2D Rad Max table.

    This is not directly a IRF component but is needed as additional information
    for point-like IRF components when an energy or field of view
    dependent directional cut has been applied.

    Data format specification: :ref:`gadf:rad_max_2d`

    Parameters
    ----------
    energy_axis : `MapAxis`
        Reconstructed energy axis
    offset_axis : `MapAxis`
        Field of view offset axis.
    data : `~astropy.units.Quantity`
        Applied directional cut
    meta : dict
        Meta data
    """

    tag = "rad_max_2d"
    required_axes = ["energy", "offset"]
