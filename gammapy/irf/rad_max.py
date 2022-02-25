import astropy.units as u

from .core import IRF

__all__ = [
    "RadMax2D",
]


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
    default_unit = u.deg


    @classmethod
    def from_irf(cls, irf):
        '''
        Create a RadMax2D instance from another IRF component.

        This reads the RAD_MAX metadata keyword from the irf and creates
        a RadMax2D with a single bin in energy and offset using the
        ranges from the input irf.

        Parameters
        ----------
        irf: `~gammapy.irf.EffectiveAreaTable2D` or `~gammapy.irf.EnergyDispersion2D`
            IRF instance from which to read the RAD_MAX and limit information

        Returns
        -------
        rad_max: `RadMax2D`
            `RadMax2D` object with a single bin corresponding to the fixed
            RAD_MAX cut.

        Notes
        -----
        This assumes the true energy axis limits are also valid for the
        reco energy limits.
        '''
        if not irf.is_pointlike:
            raise ValueError('RadMax2D.from_irf is only valid for point-like irfs')

        if 'RAD_MAX' not in irf.meta:
            raise ValueError('irf does not contain RAD_MAX keyword')

        rad_max_value = irf.meta["RAD_MAX"]
        if not isinstance(rad_max_value, float):
            raise ValueError('RAD_MAX must be a float')

        energy_axis = irf.axes["energy_true"].copy(name="energy").squash()
        offset_axis = irf.axes["offset"].squash()

        return cls(
            data=u.Quantity([[rad_max_value]], u.deg),
            axes=[energy_axis, offset_axis],
        )
