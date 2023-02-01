# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np


def get_irfs_features(
    observations,
    coord,
    names=["edisp-bias", "edisp-res", "psf-radius"],
    containment_fraction=0.68,
):
    """Get features from irfs properties at a given position.
    Used for observations clustering.

    Parameters
    ----------
    coord : `~gammapy.maps.MapCoord`
        Coordinate in lon, lat, energy_true to evaluate the IRFs.
    names : list of str
        IRFs properties to be considered.
        Available options are ["edisp-bias", "edisp-res", "psf-radius"]
        (all used by default).
    containment_fraction : float
        Containment_fraction to compute the `psf-radius`.
        Default is 68%.

    Returns
    -------
    features : array
        Features

    """

    n_obs = len(observations)
    n_features = len(names)
    features = np.zeros((n_obs, n_features))
    for (
        ko,
        obs,
    ) in enumerate(observations):
        psf_kwargs = dict(fraction=containment_fraction, energy_true=energy_true)
        if isinstance(obs.psf, PSFMap) and isinstance(obs.edisp, EDispKernelMap):
            edisp_kernel = obs.edisp.get_edisp_kernel(position=coord.skycoord[0])
            psf_kwargs["position"] = coord.skycoord[0]
        else:
            offset_max = np.minimum(
                obs.psf.axes["offset"].center[-1], obs.edisp.axes["offset"].center[-1]
            )
            offset = np.minimum(
                coord.skycoord.separation(obs.pointing_radec)[0], offset_max
            )
            edisp_kernel = obs.edisp.to_edisp_kernel(offset)
            psf_kwargs["offset"] = offset
        for kf, name in enumerate(names):
            if name == "edisp-bias":
                features[ko, kf] = edisp_kernel.get_bias(energy_true)
            if name == "edisp-res":
                features[ko, kf] = edisp_kernel.get_resolution(energy_true)
            if name == "psf-radius":
                features[ko, kf] = obs.psf.containment_radius(**psf_kwargs).value
    return features
