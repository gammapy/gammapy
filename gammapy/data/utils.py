# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np


def get_irfs_features(
    observations,
    coord,
    names=["edisp-bias", "edisp-res", "psf-radius"],
    containment_faction=0.68,
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
    containment_faction : float
        Containment_faction to compute the `psf-radius`.
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
        offset_max = np.minimum(
            obs.psf.axes["offset"].center[-1], obs.edisp.axes["offset"].center[-1]
        )
        for kf, name in enumerate(names):
            offset = np.minimum(
                coord.skycoord.separation(obs.pointing_radec)[0], offset_max
            )
            energy_true = coord["energy_true"][0]
            edisp_kernel = obs.edisp.to_edisp_kernel(offset)
            if name == "edisp-bias":
                features[ko, kf] = edisp_kernel.get_bias(energy_true)
            if name == "edisp-res":
                features[ko, kf] = edisp_kernel.get_resolution(energy_true)
            if name == "psf-radius":
                psf_radius = obs.psf.containment_radius(
                    fraction=containment_faction,
                    offset=offset,
                    energy_true=energy_true,
                )
                features[ko, kf] = psf_radius.value
    return features
