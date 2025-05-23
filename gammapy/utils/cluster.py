# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for hierarchical/agglomerative clustering."""
import numpy as np
import scipy.cluster.hierarchy as sch

__all__ = ["standard_scaler", "hierarchical_clustering"]


def standard_scaler(features):
    r"""Compute standardized features by removing the mean and scaling to unit variance.

    Calculated through:

    .. math::
        f_\text{scaled} = \frac{f-\text{mean}(f)}{\text{std}(f)} .

    Parameters
    ----------
    features : `~astropy.table.Table`
        Table containing the features.

    Returns
    -------
    scaled_features : `~astropy.table.Table`
        Table containing the scaled features (dimensionless).


    Examples
    --------
    Compute standardized features of a cluster observations based on their IRF quantities::

    >>> from gammapy.data.utils import get_irfs_features
    >>> from gammapy.data import DataStore
    >>> from gammapy.utils.cluster import standard_scaler
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u

    >>> data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    >>> obs_ids = data_store.obs_table["OBS_ID"][:3]
    >>> obs = data_store.get_observations(obs_ids)

    >>> position = SkyCoord(329.716 * u.deg, -30.225 * u.deg, frame="icrs")
    >>> names = ["edisp-res", "psf-radius"]
    >>> features_irfs = get_irfs_features(
    ...     obs,
    ...     energy_true="1 TeV",
    ...     position=position,
    ...     names=names
    ... )
    >>> scaled_features_irfs = standard_scaler(features_irfs)
    >>> print(scaled_features_irfs)
         edisp-res      obs_id      psf-radius
    ------------------- ------ --------------------
    -0.1379190199428797  20136 -0.18046952655570045
     1.2878662980210884  20137   1.3049664466089965
    -1.1499472780781963  20151  -1.1244969200533408
    """
    scaled_features = features.copy()
    for col in scaled_features.columns:
        if col not in ["obs_id", "dataset_name"]:
            data = scaled_features[col].data
            scaled_features[col] = (data - data.mean()) / data.std()
    return scaled_features


def hierarchical_clustering(features, linkage_kwargs=None, fcluster_kwargs=None):
    """Hierarchical clustering using given features.

    Parameters
    ----------
    features : `~astropy.table.Table`
        Table containing the features.
    linkage_kwargs : dict, optional
        Arguments forwarded to `scipy.cluster.hierarchy.linkage`.
        Default is None, which uses method="ward" and metric="euclidean".
    fcluster_kwargs : dict, optional
        Arguments forwarded to `scipy.cluster.hierarchy.fcluster`.
        Default is None, which uses criterion="maxclust" and t=3.


    Returns
    -------
    features : `~astropy.table.Table`
        Table containing the features and an extra column for the groups labels.


    Examples
    --------
    Cluster features into t=2 groups with a corresponding label for each group::

    >>> from gammapy.data.utils import get_irfs_features
    >>> from gammapy.data import DataStore
    >>> from gammapy.utils.cluster import standard_scaler, hierarchical_clustering
    >>> from astropy.coordinates import SkyCoord
    >>> import astropy.units as u

    >>> data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1/")
    >>> obs_ids = data_store.obs_table["OBS_ID"][13:20]
    >>> obs = data_store.get_observations(obs_ids)

    >>> position = SkyCoord(329.716 * u.deg, -30.225 * u.deg, frame="icrs")
    >>> names = ["edisp-res", "psf-radius"]
    >>> features_irfs = get_irfs_features(
    ...     obs,
    ...     energy_true="1 TeV",
    ...     position=position,
    ...     names=names
    ... )
    >>> scaled_features_irfs = standard_scaler(features_irfs)

    >>> features = hierarchical_clustering(scaled_features_irfs, fcluster_kwargs={"t": 2})
    >>> print(features)
         edisp-res      obs_id      psf-radius     labels
    ------------------- ------ ------------------- ------
    -1.3020791585772495  20326 -1.2471938975366008      2
    -1.3319831545301117  20327 -1.4586649826004114      2
    -0.7763307219821931  20339 -0.6705024680435898      2
     0.9677107409819438  20343  0.9500979841335693      1
      0.820562952023891  20344  0.8160964882165554      1
     0.7771617763704126  20345  0.7718272408581743      1
     0.8449575657133206  20346  0.8383396349722769      1

    """
    features = features.copy()
    features_array = np.array(
        [
            features[col].data
            for col in features.columns
            if col not in ["obs_id", "dataset_name"]
        ]
    ).T

    default_linkage_kwargs = dict(method="ward", metric="euclidean")
    if linkage_kwargs is not None:
        default_linkage_kwargs.update(linkage_kwargs)

    pairwise_distances = sch.distance.pdist(features_array)
    linkage = sch.linkage(pairwise_distances, **default_linkage_kwargs)

    default_fcluster_kwargs = dict(criterion="maxclust", t=3)
    if fcluster_kwargs is not None:
        default_fcluster_kwargs.update(fcluster_kwargs)
    labels = sch.fcluster(linkage, **default_fcluster_kwargs)

    features["labels"] = labels
    return features
