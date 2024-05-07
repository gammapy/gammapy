# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.table import Table
from gammapy.utils.cluster import standard_scaler

__all__ = ["get_irfs_features"]


def get_irfs_features(
    observations,
    energy_true,
    position=None,
    fixed_offset=None,
    names=None,
    containment_fraction=0.68,
    apply_standard_scaler=False,
):
    """Get features from IRFs properties at a given position. Used for observations clustering.

    Parameters
    ----------
    observations : `~gammapy.data.Observations`
        Container holding a list of `~gammapy.data.Observation`.
    energy_true : `~astropy.units.Quantity`
        Energy true at which to compute the containment radius.
    position : `~astropy.coordinates.SkyCoord`, optional
        Sky position. Default is None.
    fixed_offset : `~astropy.coordinates.Angle`, optional
        Offset calculated from the pointing position. Default is None.
        If neither the `position` nor the `fixed_offset` is specified,
        it uses the position of the center of the map by default.
    names : {"edisp-bias", "edisp-res", "psf-radius"}
        IRFs properties to be considered.
        Default is None. If None, all the features are computed.
    containment_fraction : float, optional
        Containment_fraction to compute the `psf-radius`.
        Default is 68%.
    apply_standard_scaler : bool, optional
        Compute standardize features by removing the mean and scaling to unit variance.
        Default is False.

    Returns
    -------
    features : `~astropy.table.Table`
        Features table.

    Examples
    --------
    Compute the IRF features for "edisp-bias", "edisp-res" and "psf-radius" at 1 TeV::

    >>> from gammapy.data.utils import get_irfs_features
    >>> from gammapy.data import DataStore
    >>> from gammapy.utils.cluster import standard_scaler
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u

    >>> selection = dict(
    ...     type="sky_circle",
    ...     frame="icrs",
    ...     lon="329.716 deg",
    ...     lat="-30.225 deg",
    ...     radius="2 deg",
    ... )

    >>> data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    >>> obs_table = data_store.obs_table.select_observations(selection)
    >>> obs = data_store.get_observations(obs_table["OBS_ID"][:3])

    >>> position = SkyCoord(329.716 * u.deg, -30.225 * u.deg, frame="icrs")
    >>> names = ["edisp-bias", "edisp-res", "psf-radius"]
    >>> features_irfs = get_irfs_features(
    ...     obs,
    ...     energy_true="1 TeV",
    ...     position=position,
    ...     names=names,
    ... )

    >>> print(features_irfs)
         edisp-bias     obs_id      edisp-res           psf-radius
                                                       deg
        ------------------- ------ ------------------- -------------------
        0.11587179071752986  33787   0.368346217294295 0.14149953611195087
        0.04897634344908595  33788 0.33983991887701287 0.11553325504064559
          0.033176650892097  33789 0.32377509405904137 0.10262943822890519

    """
    from gammapy.irf import EDispKernelMap, PSFMap

    if names is None:
        names = ["edisp-bias", "edisp-res", "psf-radius"]

    if position and fixed_offset:
        raise ValueError(
            "`position` and `fixed_offset` arguments are mutually exclusive"
        )

    rows = []

    for obs in observations:
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
                        position.separation(obs.get_pointing_icrs(obs.tmid)), offset_max
                    )
            else:
                offset = fixed_offset
            edisp_kernel = obs.edisp.to_edisp_kernel(offset)
            psf_kwargs["offset"] = offset

        data = {}
        for name in names:
            if name == "edisp-bias":
                data[name] = edisp_kernel.get_bias(energy_true)[0]
            if name == "edisp-res":
                data[name] = edisp_kernel.get_resolution(energy_true)[0]
            if name == "psf-radius":
                data[name] = obs.psf.containment_radius(**psf_kwargs).to("deg")
            data["obs_id"] = obs.obs_id

        rows.append(data)

    features = Table(rows)

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
