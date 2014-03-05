import numpy as np
from group import *
from sherpa_contrib.profiles import *
from sherpa.astro.ui import *


class spectral_data():
    """ Class to encapsulate spectral data and background to be used with wstat
        its is created from sherpa data and background
    """
    def __init__(self,data,bkg):
        self.bkg = bkg
        self.data = data
        self.ONexpo = data.exposure
        self.ONalpha = data.backscal
        self.OFFexpo = bkg.exposure
        self.OFFalpha = bkg.backscal
        self.alpha = self.ONexpo/self.OFFexpo*self.ONalpha/self.OFFalpha
        self.Ttot = self.ONexpo *(1+ 1./self.alpha)#+ self.OFFexpo
        self.rebin()

    def rebin(self):
        index_notice=np.int_(self.bkg.get_noticed_channels()-1)
        bkg_grp=grpGetGroupSum(self.bkg.counts[index_notice],self.bkg.grouping[index_notice])
        f = self.bkg.grouping[index_notice] == 1
        self.fit_bkg = bkg_grp[f]
        self.fit_alpha = np.zeros(self.fit_bkg.shape) + self.alpha
        self.fit_Ttot = np.zeros(self.fit_bkg.shape) + self.Ttot
        self.fit_ONexpo = np.zeros(self.fit_bkg.shape) + self.ONexpo
        
       
class w_statistic(object):
    """ A tentative approach to deal with the wstat in sherpa
        to compute the proper statistics one needs to use 
        the signal and background datasets as well as the model 
    """

    def __init__(self, dataids):
        self.datasets = []
        for dataid in dataids:
            spec = spectral_data(get_data(dataid),get_bkg(dataid))
            self.datasets.append(spec)
        self.bkg = np.concatenate([ a.fit_bkg for a in self.datasets ])
        self.alpha = np.concatenate([ a.fit_alpha for a in self.datasets ])
        self.Ttot = np.concatenate([ a.fit_Ttot for a in self.datasets ])
        self.ONexpo = np.concatenate([ a.fit_ONexpo for a in self.datasets ])


    def __call__(self,data,model,staterror=None, syserror=None,weight=None):
        """ The actual call of the wstat function
            returns a modified model that can be fitted 
            with the usual cash statistic
        """
        trunc_val = 1e-25

        backg = self.bkg 
        mod = model/self.ONexpo # count rate
        # Compute correction factors for the likelihood function
        afactor = np.sqrt((self.Ttot*mod-data-backg)**2+4*self.Ttot*backg*mod)
        bfactor = 0.5*(data+backg-self.Ttot*mod+afactor)/self.Ttot

        # Final expected bkg in ON dataset 
        modelON = model+bfactor*self.ONexpo
        # Expected bkg in OFF dataset
        modelOFF = bfactor/self.alpha*self.ONexpo

        # to avoid issues with log 0
        modelON[modelON<=0.0] = trunc_val
        modelOFF[modelOFF<=0.0] = trunc_val
        data[data<=0.0] = trunc_val
        backg[backg<=0.0] = trunc_val

        # Poisson log-likelihood with background variation using normalization to provide pseudo goodness-of-fit (a la cstat) 
        likelihood = data*np.log(modelON) + backg*np.log(modelOFF)+data*(1-np.log(data)) +backg*(1-np.log(backg))-modelON-modelOFF 
        # reverse sign to minimize log-likelihood!
        likelihood *= -2.0
        stat = np.sum(likelihood)
 #       print stat, model.sum(), modelON.sum(), modelOFF.sum(),backg.sum(), data.sum()
        return (stat,np.sqrt(data+self.alpha**2*backg))

    # This is apparently required although this could be set to None
    def CATstat_err_LV(self,data):
        return np.ones_like(data)

def wfit(dataids):
    set_method("neldermead")
    wstat = w_statistic(dataids)
    load_user_stat("mystat", wstat,wstat.CATstat_err_LV)
    set_stat(mystat)
    fit(*dataids)






