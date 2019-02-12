from astropy.utils import lazyproperty
from .parameter import Parameters


class Datasets:
    """Join multiple datasets

    Parameters
    ----------
    datasets : `Dataset` or list of `Dataset`
        List of `Dataset` objects ot be joined.

    """

    def __init__(self, datasets, mask=None):
        if not isinstance(datasets, list):
            self.datasets = [datasets]
        self.datasets = datasets
        self.mask = mask

    @lazyproperty
    def is_same_type(self):
        """Whether all contained datasets are of the same type"""
        dataset_type = type(self.datasets[0])
        is_dataset_type = [isinstance(_, dataset_type) for _ in self.datasets]
        return np.all(is_dataset_type)

    @property
    def parameters(self):
        return Parameters([dataset.parameters for dataset in self.datasets])

    def likelihood_per_bin(self):
        """"""
        if not self.is_same_type:
            raise ValueError("Cannot join likelihood per bin, if datasets are not of the same type")

        for idx, dataset in enumerate(self.datasets):
            if idx == 0:
                total_likelihood = dataset.likelihood_per_bin(None)
            else:
                total_likelihood += dataset.likelihood_per_bin(None)

        return total_likelihood


    def _likelihood(self, parameters=None):
        """Compute joint likelihood"""
        total_likelihood = 0
        # TODO: add parallel evaluation of likelihoods
        for dataset in self.datasets:
            total_likelihood += dataset.likelihood(None)
        return total_likelihood


    def _likelihood_same_type(self, parameters):
        """Total likelihood given the current model parameters"""
        # update parameters
        if self.mask:
            stat = self.likelihood_per_bin()[self.mask]
        else:
            stat = self.likelihood_per_bin()
        return np.nansum(stat, dtype=np.float64)