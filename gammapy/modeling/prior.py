from . import Parameter


class Prior:
    """
    Base prior class containing methods to serialise.
    """

    def __call__(self, base):
        if isinstance(base, Parameter):
            return self.evaluate(base.value)
        else:  # model-like instance
            return self.evaluate(base.parameters.value)

    def to_dict(self):
        output = {
            "tag": self.tag[0],
        }
        output["parameters"] = [
            {"name": p[0], "value": p[1]} for p in self.prior_parameters
        ]
        return output

    def __str__(self):
        string = f"{self.__class__.__name__}\n"
        string += "-" * 10
        for p in self.prior_parameters:
            string += f"\n{p[0]:10}: {p[1]}"
        return string


class GaussianPrior(Prior):
    """Gaussian Prior with mu and sigma."""

    tag = ["GaussianPrior"]

    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def evaluate(self, value):
        return ((value - self.mu) / self.sigma) ** 2

    @property
    def prior_parameters(self):
        return [("mu", self.mu), ("simga", self.sigma)]


class UniformPrior(Prior):
    """Uniform Prior"""

    tag = ["UniformPrior"]

    def __init__(self, uni=0):
        self.uni = uni

    def evaluate(self, value):
        return self.uni

    @property
    def prior_parameters(self):
        return [("uni", self.uni)]


class MultivariateGaussianPrior(Prior):
    """Gaussian Priors on mulitple parameters with different mu and sigma. Set on a model.

    Parameters
    ----------
    mus : `~numpy.ndarray`
        Array with the expected mean of the parameters in the same order as the to be evaluated parameters.
    sigmas : `~numpy.ndarray`
        Array with the expected standard deviation of the parameters in the same order as the to be evaluated parameters.
    """

    tag = ["MultivariateGaussianPrior"]

    def __init__(self, mus, sigmas):
        self.mus = mus
        self.sigmas = sigmas

    def evaluate(self, values):
        return ((values - self.mus) / self.sigmas) ** 2

    @property
    def prior_parameters(self):
        return [("mus", self.mus), ("sigmas", self.sigmas)]


class CovarianceGaussianPrior(Prior):
    """Gaussian Priors on mulitple parameters with different mu and sigma. Set on a model.

    Parameters
    ----------
    cov : `~numpy.ndarray`
        Covariance matrix in the same order as the to be evaluated parameters.
    """

    tag = ["CovarianceGaussianPrior"]

    def __init__(self, cov):
        self.cov = cov

    def evaluate(self, values):
        from numpy import linalg

        return values @ linalg.inv(self.cov) @ values

    @property
    def prior_parameters(self):
        return [("cov", self.cov)]
