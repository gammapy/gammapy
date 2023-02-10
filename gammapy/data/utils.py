# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.table import Column, Table
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.utils.cluster import standard_scaler


def get_irfs_features(
    observations,
    energy_true,
    position=None,
    fixed_offset=None,
    names=None,
    containment_fraction=0.68,
    apply_standard_scaler=False,
):
    """Get features from irfs properties at a given position.
    Used for observations clustering.

    Parameters
    ----------
    observations : `~gammapy.data.Observations`
        Container holding a list of `~gammapy.data.Observation`
    energy_true : `~astropy.units.Quantity`
        Energy true at which to compute the containment radius
    position : `~astropy.coordinates.SkyCoord`
        Sky position.
    fixed_offset : `~astropy.coordinates.Angle`
        Offset calculated from the pointing position.
        If neither the position nor fixed_offset is specified,
        it uses the position of the center of the map by default.
    names : list of str
        IRFs properties to be considered.
        Available options are ["edisp-bias", "edisp-res", "psf-radius"]
        (all used by default).
    containment_fraction : float
        Containment_fraction to compute the `psf-radius`.
        Default is 68%.
    standard_scaler : bool
        Compute standardize features by removing the mean and scaling to unit variance.
        Default is False.

    Returns
    -------
    features : `~astropy.table.Table`
        Features table

    """

    if names is None:
        names = ["edisp-bias", "edisp-res", "psf-radius"]

    if position and fixed_offset:
        raise ValueError(
            "`position` and `fixed_offset` arguments are mutually exclusive"
        )

    n_obs = len(observations)
    n_features = len(names)
    data = np.zeros((n_obs, n_features))
    units = [u.Unit("")] * n_features
    for (
        ko,
        obs,
    ) in enumerate(observations):
        psf_kwargs = dict(fraction=containment_fraction, energy_true=energy_true)
        if isinstance(obs.psf, PSFMap) and isinstance(obs.edisp, EDispKernelMap):
            if position is None:
                position = obs.psf.psf_map.geom.center_skydir
            edisp_kernel = obs.edisp.get_edisp_kernel(position=position)
            psf_kwargs["position"] = position
        else:
            if fixed_offset is None:
                if position is None:
                    offset = 0 * u.deg
                else:
                    offset_max = np.minimum(
                        obs.psf.axes["offset"].center[-1],
                        obs.edisp.axes["offset"].center[-1],
                    )
                    offset = np.minimum(
                        position.separation(obs.pointing_radec), offset_max
                    )
            else:
                offset = fixed_offset
            edisp_kernel = obs.edisp.to_edisp_kernel(offset)
            psf_kwargs["offset"] = offset
        for kf, name in enumerate(names):
            if name == "edisp-bias":
                data[ko, kf] = edisp_kernel.get_bias(energy_true)
            if name == "edisp-res":
                data[ko, kf] = edisp_kernel.get_resolution(energy_true)
            if name == "psf-radius":
                containment_radius = obs.psf.containment_radius(**psf_kwargs).to("deg")
                data[ko, kf] = containment_radius.value
                units[kf] = u.deg

    features = Table(data, names=names, units=units)
    features.add_column(Column(observations.ids, name="obs_id"), index=0)

    if apply_standard_scaler:
        features = standard_scaler(features)

    features.meta = dict(
        energy_true=energy_true,
        fixed_offset=fixed_offset,
        containment_fraction=containment_fraction,
        apply_standard_scaler=apply_standard_scaler,
    )

    if position:
        features.meta["lon"] = position.galactic.l
        features.meta["lat"] = position.galactic.b
        features.meta["frame"] = "galactic"

    return features
