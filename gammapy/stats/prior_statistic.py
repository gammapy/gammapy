class PriorFitStatistic:
    """Class to compute statistics due to the priors set on models and their parameters.

    Parameters
    ----------
    weight : float
        Weight of the prior
    """

    def __init__(self, weight=1):
        self.weight = weight

    def stat_sum(self, models):
        stat = 0
        for m in models:
            if m.prior is not None:
                stat += m.prior_stat_sum()
            for par in m.parameters:
                if par.prior is not None:
                    stat += par.prior_stat_sum()
        return self.weight * stat
