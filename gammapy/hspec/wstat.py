# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from group import grpGetGroupSum
import sherpa.astro.ui as sau


class spectral_data(object):
    """Class to encapsulate spectral data and background to be used with wstat
    its is created from sherpa data and background.
    """

    def __init__(self, data, bkg):
        self.bkg = bkg
        self.data = data
        self.ONexpo = data.exposure
        self.ONalpha = data.backscal
        self.OFFexpo = bkg.exposure
        self.OFFalpha = bkg.backscal
        self.alpha = self.ONexpo / self.OFFexpo * self.ONalpha / self.OFFalpha
        self.Ttot = self.ONexpo * (1 + 1. / self.alpha)  # + self.OFFexpo
        self.rebin()

    def rebin(self):
        index_notice = np.int_(self.data.get_noticed_channels() - 1)
        if self.data.grouping is not None:
            bkg_grp = grpGetGroupSum(self.bkg.counts[index_notice],
                                     self.data.grouping[index_notice])
            f = self.data.grouping[index_notice] == 1  # ??
            self.fit_bkg = bkg_grp[f]

            dat_grp = grpGetGroupSum(self.data.counts[index_notice],
                                     self.data.grouping[index_notice])
            self.fit_data = dat_grp[f]

        else:
            self.fit_bkg = self.bkg.counts[index_notice]
            self.fit_data = self.data.counts[index_notice]

        # WARNING this is dangerous since we implicitly assume there is no grouping <-- THIS MAY NOT BE THE CASE
        self.excess = np.zeros_like(self.data.counts)
        self.excess[index_notice] = self.fit_data - self.alpha * self.fit_bkg
        self.full_expo = np.zeros_like(self.data.counts)
        self.full_expo[index_notice] = self.ONexpo

        self.fit_alpha = np.zeros(self.fit_bkg.shape) + self.alpha
        self.fit_Ttot = np.zeros(self.fit_bkg.shape) + self.Ttot
        self.fit_ONexpo = np.zeros(self.fit_bkg.shape) + self.ONexpo


class w_statistic(object):
    """A tentative approach to deal with the wstat in sherpa
    to compute the proper statistics one needs to use
    the signal and background datasets as well as the model.
    """

    def __init__(self, dataids):
        self.datasets = []
        self.tot_excess = None
        self.tot_expo = None

        for dataid in dataids:
            spec = spectral_data(sau.get_data(dataid), sau.get_bkg(dataid))
            self.datasets.append(spec)
        self.bkg = np.concatenate([a.fit_bkg for a in self.datasets])
        self.alpha = np.concatenate([a.fit_alpha for a in self.datasets])
        self.Ttot = np.concatenate([a.fit_Ttot for a in self.datasets])
        self.ONexpo = np.concatenate([a.fit_ONexpo for a in self.datasets])

        for a in self.datasets:
            # Carefull we are assuming that all the spectra have the same binning
            if self.tot_excess is None:
                self.tot_excess = np.zeros_like(a.excess)
            if self.tot_expo is None:
                self.tot_expo = np.zeros_like(a.excess)

            self.tot_excess += a.excess
            self.tot_expo += a.full_expo

    def __call__(self, data, model, staterror=None, syserror=None, weight=None):
        """The actual call of the wstat function
        returns a modified model that can be fitted
        with the usual cash statistic.
        """

        backg = self.bkg
        mod = model / self.ONexpo  # count rate

        # Compute correction factors for the likelihood function
        afactor = np.sqrt((self.Ttot * mod - data - backg) ** 2 + 4 * self.Ttot * backg * mod)
        bfactor = 0.5 * (data + backg - self.Ttot * mod + afactor) / self.Ttot

        # Final expected bkg in ON dataset 
        modelON = model + bfactor * self.ONexpo
        # Expected bkg in OFF dataset
        modelOFF = bfactor / self.alpha * self.ONexpo

        # to avoid issues with log 0
        trunc_val = 1e-20
        modelON[modelON <= 0.0] = trunc_val
        modelOFF[modelOFF <= 0.0] = trunc_val
        data[data <= 0.0] = trunc_val
        backg[backg <= 0.0] = trunc_val

        # Poisson log-likelihood with background variation using normalization to provide pseudo goodness-of-fit (a la cstat) 
        likelihood = data * np.log(modelON) + backg * np.log(modelOFF) + data * (1 - np.log(data)) + backg * (
            1 - np.log(backg)) - modelON - modelOFF
        # reverse sign to minimize log-likelihood!
        likelihood *= -2.0
        stat = np.sum(likelihood)
        # print(stat, model.sum(), modelON.sum(), modelOFF.sum(),backg.sum(), data.sum())
        return (stat, np.sqrt(data + self.alpha ** 2 * backg))

    # This is apparently required although this could be set to None
    def CATstat_err_LV(self, data):
        return np.ones_like(data)


def wfit(dataids=None):
    listids = ()
    if dataids is None:
        listids = sau.list_data_ids()
    else:
        listids = dataids

    wstat = w_statistic(listids)
    sau.load_user_stat("mystat", wstat, wstat.CATstat_err_LV)
    sau.set_stat(mystat)
    sau.set_method("neldermead")  # set_method("moncar")
    sau.set_conf_opt("max_rstat", 1000)  # We don't use a specific maximum reduced statistic value
    # since we don't expect the cstat to be anywhere near the
    # large number limit
    sau.fit(*listids)

    sau.conf()
