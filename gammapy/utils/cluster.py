# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils for hierarchical/agglomerative clustering."""

import scipy.cluster.hierarchy as sch


def hierarchical_clustering(
    features, linkage_kwargs=None, fcluster_kwargs=None, standard_scaler=False
):
    """Hierarchical clustering using given features.

    Parameters
    ----------
    features : array
        (N x M) array with N observations and M features.
    linkage_kwargs : dict
        Arguments forwarded to `scipy.cluster.hierarchy.linkage`
    fcluster_kwargs : dict
        Arguments forwarded to `scipy.cluster.hierarchy.fcluster`
    standard_scaler : bool
        Standardize features by removing the mean and scaling to unit variance.
        Default is False.

    Returns
    -------
    ind_clusters : array
        Array of cluster ID
    features : array
        Scaled features
    """

    if standard_scaler:
        features = features.copy()
        for kf in range(features.shape[1]):
            features[:, kf] = (features[:, kf] - features[:, kf].mean()) / features[
                :, kf
            ].std()

    default_linkage_kwargs = dict(method="ward", metric="euclidean")
    if linkage_kwargs is not None:
        default_linkage_kwargs.update(linkage_kwargs)

    pairwise_distances = sch.distance.pdist(features)
    linkage = sch.linkage(pairwise_distances, **default_linkage_kwargs)

    default_fcluster_kwargs = dict(criterion="maxclust", t=3)
    if fcluster_kwargs is not None:
        default_fcluster_kwargs.update(fcluster_kwargs)
    ind_clusters = sch.fcluster(linkage, **default_fcluster_kwargs)

    return ind_clusters, features
