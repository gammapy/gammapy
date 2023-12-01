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
