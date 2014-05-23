import numpy as np
from sherpa_contrib.profiles import *
from sherpa.astro.ui import *


class wstat_2D(object):
    """ A tentative approach to deal with the wstat in sherpa
        to compute the proper statistics one needs to use 
        the signal and background datasets as well as the model 
    """

    def __init__(self, ONmap,OFFmap,alphamap):
        self.ONmap = ONmap
        self.OFFmap = OFFmap
        self.alpha = alphamap

    def __call__(self,data,model,staterror=None, syserror=None,weight=None):
        """ The actual call of the wstat function
            returns a modified model that can be fitted 
            with the usual cash statistic
        """
        trunc_val = 1e-25
        ON = self.ONmap
        OFF = self.OFFmap 
        
        TOT = ON + OFF
        corr = self.alpha/(self.alpha+1.0)

        # Compute correction factors for the likelihood function
        afactor = np.sqrt((model-TOT*corr)**2+4*OFF*model*corr)
        bfactor = 0.5*(TOT*corr-model+afactor)

        # Final expected bkg in ON dataset 
        modelON = model+bfactor
        # Expected bkg in OFF dataset
        modelOFF = bfactor/self.alpha

        # to avoid issues with log 0
        modelON[modelON<=0.0] = trunc_val
        modelOFF[modelOFF<=0.0] = trunc_val
 #       modelOFF[np.isinf(modelOFF)] = trunc_val 
        ON[ON<=0.0] = trunc_val
        OFF[OFF<=0.0] = trunc_val

        # Poisson log-likelihood with background variation using normalization to provide pseudo goodness-of-fit (a la cstat) 
        likelihood = ON*np.log(modelON) + OFF*np.log(modelOFF) + ON*(1-np.log(ON)) + OFF*(1-np.log(OFF))-modelON-modelOFF 

        # reverse sign to minimize log-likelihood!
        likelihood *= -2.0
        stat = np.sum(likelihood)
        #print stat, model.sum(), bfactor.sum(), modelON.sum(), ON.sum(), modelOFF.sum(),OFF.sum()
        return (stat,np.sqrt(ON+self.alpha**2*OFF))

    # This is apparently required although this could be set to None
    def CATstat_err_LV(self,data):
        return np.ones_like(data)

"""
def wfit(dataids):
    set_method("neldermead")
    wstat = w_statistic()
    load_user_stat("mystat", wstat2D,wstat2D.CATstat_err_LV)
    set_stat(mystat)
    fit(*dataids)
"""





