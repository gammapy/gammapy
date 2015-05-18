from sherpa.astro.ui import load_pha,get_data,notice_id,set_source,covar,conf,set_conf_opt,copy_data,set_full_model,get_fit_results,get_conf_results
import wstat
import numpy as np
import copy
from sherpa.astro.data import DataPHA

d2r = np.pi/180.

def print_fit():
    res = get_fit_results()
    print "Fit success : \t", res.succeeded
    print "Fit results:"
    print "Numpoints :\t", res.numpoints
    print "dof :\t\t", res.dof
    print "Final statistic value : \t", res.statval
    print "Fitted parameters:"
    for index, parn in enumerate(res.parnames):
        print parn, "\t\t{0:.3e}".format(res.parvals[index]) 

def print_conf():
    res = get_conf_results()
    print "Confidence limits on fitted parameters:"
    for index, parn in enumerate(res.parnames):
        print parn, "\t\t{0:.3e}".format(res.parvals[index]), "\t{0:.3e}".format(res.parmins[index]),"\t{0:.3e}".format(res.parmaxes[index]) 
   
    

class HESS_spec():
    """ Class to encapsulate HESS specific data for spectral analysis with sherpa
    """
    def __init__(self,name,filename=None):
        self.name = name
        if filename is not None:
#            print filename
            load_pha(name,filename)
            self.data=get_data(name)
            self.arf=self.data.get_arf()
            self.rmf=self.data.get_rmf()

            # Read keywords from pha header
            try:
                self.threshold = self.data.header['ETH']
            except KeyError:
                print "WARNING: no threshold found using 200 GeV"
                self.threshold = 2e8   # default value 200 GeV
                self.emax = 1e11           # default value 100 TeV
                
            try:
                self.zenith = self.data.header['ZENITH']
            except KeyError:
                    print "WARNING: no mean zenith angle found using 45 deg"
                    self.zenith = 45.0   # default value 45 deg
 
            try:
                self.offset = self.data.header['OFFSET']
            except KeyError:
                print "WARNING: no offset angle found using 1.0 deg"
                self.offset = 1.0   # default value 1 deg

            try:
                self.n_tels = self.data.header['N_TELS']
            except KeyError:
                print "WARNING: no number of telescopes found using 0"
                self.n_tels = 0   # default value 

            try:
                self.eff = self.data.header['EFFICIEN']
            except KeyError:
                print "WARNING: no efficiency found using 1.0"
                self.eff = 1.00   # default value 

            try:
                self.tstart = self.data.header['TSTART']
            except KeyError:
                print "WARNING: no tstart found using 0"
                self.tstart = 0.   # default value 

            try:
                self.tstop = self.data.header['TSTOP']
            except KeyError:
                print "WARNING: no tstop found using tsart+1800"
                self.tstop = self.tstart+1800   # default value 

        else:
            self.data=get_data(name)
            self.arf=self.data.get_arf()
            self.rmf=self.data.get_rmf()
           

    def set_threshold(self,thres_val):
        self.threshold = thres_val
        
    def set_emax(self,emax):
        self.emax = emax

    def notice(self,min_ener,max_ener):
        """ Notice energy range 
            if minimal value required below threshold, use threshold instead
            if maximal value required beyond emax, use emax instead
        """    
        self.data.notice(max(self.threshold,min_ener),min(self.emax,max_ener))

    def set_source(self,model):
        """ Apply source model to the dataset """
        set_source(self.name,model)

    def set_minimalArea(self,areamin):
        """ Define threshold using minimal area
            Extract true energy value from arf file
            To be implemented
            """
        my_arf = get_arf(self.name)





class SpecSource():
    """ Class to load and encapsulate all datasets used for a spectral analysis
    """
    def __init__(self,name,filelist=None):
        self.name = name
        
        self.listids = None
        self.noticed_ids = None
        
        if filelist is not None:
            self.loadlist(filelist)


    def loadlist(self,listfile):
        """ Load all datasets in listfile 
            expect pha files containing keywords for bkg, arf and rmf files
            store dataid
        """
        self.listids = np.empty(len(listfile),dtype=object)
        for index,filename in enumerate(listfile):
            datid = self.name+filename # temporary before I have a better idea
#            self.listids.append(HESS_spec(datid,filename))
            self.listids[index] = HESS_spec(datid,filename)
        self.noticed_ids = np.ones((len(self.listids)), dtype=bool)

        # make arrays to deal with run characteristics
        self.offsets = np.array([run.offset for run in self.listids])
        self.zeniths = np.array([run.zenith for run in self.listids])
        self.n_tels = np.array([run.n_tels for run in self.listids])
        self.thresholds = np.array([run.threshold*1e-9 for run in self.listids])     # in TeV
        self.tstarts = np.array([run.tstart for run in self.listids])
        self.tstops = np.array([run.tstop for run in self.listids])
        self.effs = np.array([run.eff for run in self.listids])
 

    """ Select runs to be used for the fit/plot procedures 
        input array gives boolean for inclusion or exclusion of run
    """
    def notice_runs(self,valid=None):
        if valid is None:
            self.noticed_ids=np.ones((len(self.listids)), dtype=bool)
        else:
            self.noticed_ids=self.noticed_ids*valid
        if self.noticed_ids.sum() == 0:
            print "Warning: noticed runs list is empty."
 
    def get_noticed_list(self):
        return [ids.name for ids in self.listids[self.noticed_ids]]
       
    """ Notice energy range in TeV. This is applied to all HESS_spec in SpecSource"""
    def notice(self,min_ener,max_ener):
        for datid in self.listids:
            datid.notice(min_ener*1e9,max_ener*1e9)

    """ Set source model. This is applied to all HESS_spec in SpecSource"""           
    def set_source(self,model):
        for datid in self.listids:
            datid.set_source(model)

    """ perform fit using profile likelihood technique for background estimation and subtraction """
    def fit(self,do_covar=False,do_conf=False):
        listnames = self.get_noticed_list()#[ids.name for ids in self.listids[self.noticed_ids]]
        if len(listnames)>0:
            wstat.wfit(listnames)
            print_fit()
            if do_covar is True:
                covar(*listnames)
            if do_conf is True:
                set_conf_opt('max_rstat', 10000)
                conf(*listnames)
                print_conf()
        else:
            print "Empty noticed runs list. No fit"
            

    def group(self,new_ext='_group',valid=None):
        """ Group spectra 
        """ 
        totON = None
        totOFF = None
        tot_time = 0.
        tot_alpha = 0.
        tot_arf = None
        tot_rmf = None
        ntrue=0
        nrec=0

        group_dat = None
        group_bkg = None
        group_arf = None
        group_rmf = None
        
        newname = self.name+new_ext

        if valid is None:
            group_ids=np.ones((len(self.listids)), dtype=bool)
        elif valid.sum()>0:   # need a better type check obviously
            group_ids = valid
        else:
            print "Empty group. Do nothing."
            return

        # loop over all datasets
        for datid in self.listids[valid]:            
            mydat = datid.data
            if totON is None:
                totON = np.zeros_like(mydat.counts)
                totOFF = np.zeros_like(mydat.get_background().counts)
                copy_data(datid.name,newname)
                group_dat = get_data(newname)
                group_dat.name= newname
                group_bkg = group_dat.get_background()
                
            # sum total ON and OFF
            totON += mydat.counts
            totOFF += mydat.get_background().counts
            
            # here we assume the background rate is the same with in each run so that we average alpha with time 
            tot_alpha += mydat.exposure/mydat.get_background_scale()
            tot_time += mydat.exposure 
            
            # compute average arf
            c_arf = mydat.get_arf().get_y()
            if tot_arf is None:
                tot_arf = np.zeros_like(c_arf)
                group_arf = group_dat.get_arf()
 
            tot_arf += c_arf*mydat.exposure 
           
            # Compute average RMF
            c_rmf = mydat.get_rmf()
            
            # for now, we assume that n_grp is always equal to 1 which is the case for HESS rmfs generated with START
            # the channels to be used in the matrix are given by the cumulative sum of n_chan  
            chans = c_rmf.n_chan.cumsum()   
               
            # if not created, instantiate tmp_rmf
            if tot_rmf is None:
                group_rmf = group_dat.get_rmf()
                ntrue = int(c_rmf.get_dims()[0])
                nrec = int(c_rmf.detchans)
                tot_rmf=np.zeros((ntrue,nrec))

            c_rmf.matrix[np.where(np.isnan(c_rmf.matrix))]=0.   
            for i in np.arange(ntrue):
                irec_lo = c_rmf.f_chan[i]
                irec_hi = c_rmf.f_chan[i]+c_rmf.n_chan[i]
                indmin = chans[i]
                indmax = chans[i]+c_rmf.n_chan[i]
                if indmax<c_rmf.matrix.shape[0]:
                    tot_rmf[i,irec_lo:irec_hi]+=c_rmf.matrix[indmin:indmax]*c_arf[i]*mydat.exposure
  
              
        tot_arf /= tot_time
        tot_arf = np.abs(tot_arf)
        for i in np.arange(nrec):
            tot_rmf[:,i] /= tot_arf*tot_time 

        tot_rmf[np.isnan(tot_rmf)]=0.
        tot_alpha = tot_time/tot_alpha 
   
        group_dat.counts = totON
        group_dat.exposure = tot_time

        group_bkg.name=newname+'_bkg'
        group_bkg.counts = totOFF
        group_bkg.backscal = 1./tot_alpha
        group_bkg.exposure = tot_time
          
        group_rmf.name=newname+'_rmf'
        (ntrue,nrec)=tot_rmf.shape
        tot_rmf = np.abs(tot_rmf)  # this is a hack and correct as long as negative elements modulus is <<1
        # reproject total rmf into new rmf with correct f_chan, n_chan and matrix
        ix,iy=np.where(tot_rmf>0.)
        tmp = np.insert(np.diff(ix),0,1)
        new_index = np.where(tmp)[0]
        
        # Get first channel for a given true energy 
        group_rmf.f_chan *= 0
        group_rmf.f_chan[ix[new_index]]=np.uint32(iy[new_index])
        
        # Find the number of channels 
        group_rmf.n_chan *= 0
        group_rmf.n_chan[ix[new_index]] = np.uint32(np.append(iy[new_index-1][1:],iy[-1])-iy[new_index]+1)
        group_rmf.matrix = tot_rmf[ix,iy]
        
        group_arf.name=newname+'_arf'
        group_arf.specresp = tot_arf

        group_dat.set_background(group_bkg)
        group_dat.set_arf(group_arf)
        group_dat.set_rmf(group_rmf)

        res = HESS_spec(newname)
        res.threshold = np.min(np.array([run.threshold for run in self.listids[valid]]))
        res.emax = 1e11 
        return res


    def reproject(self,nbins=None , maxima=None,newname=None):
        """ Function perform reprojection of inidividual spectra (i.e. per run) into bands of similar
            offset, zenith angle and efficiency
            The reprojection is performed using the group function 
            Several 
        """
        # need to add a check for the format of the nbins and maxima dictionaries
        if nbins is None:
            nbins={'offset':5,'eff':10.,'zen':10}
        if maxima is None:
            maxima={'offset':2.5,'eff':100.,'zen':70}

        off_max=maxima['offset']
        eff_max=maxima['eff']
        zen_max = maxima['zen'] * d2r
        coszen_max = np.cos(zen_max)

        n_offset_bins = nbins['offset']
        n_eff_bins =  nbins['eff']
        n_zen_bins =  nbins['zen']
                   
        zen_step = (1.0-coszen_max)/n_zen_bins
        off_step = off_max/n_offset_bins
        eff_step = eff_max/n_eff_bins

        zen_index = np.floor((np.cos(self.zeniths*d2r)-coszen_max)/zen_step).astype(int)
        off_index = np.floor(self.offsets/off_step).astype(int)
        eff_index = np.floor(self.effs/eff_step).astype(int)

        tot_index = off_index + 100*zen_index +10000*eff_index
        unique_index,array_index = np.unique(tot_index,return_inverse=True)

        # create new specsource object
        if newname is None:
            newspec = SpecSource(self.name+'_grp')
        else:
            newspec = SpecSource(newname)

        newspec.listids = np.empty(len(unique_index),dtype=object)
        newspec.noticed_ids = np.ones(len(unique_index), dtype=bool)

        newspec.offsets = np.zeros(len(unique_index))
        newspec.zeniths = np.zeros(len(unique_index))
        newspec.effs = np.zeros(len(unique_index))
        newspec.thresholds = np.zeros(len(unique_index))

        for ind in range(len(unique_index)):
            newspec.listids[ind] = self.group(new_ext='_'+str(unique_index[ind]),valid=(array_index==ind))
            #cname = self.name+'_'+str(unique_index[ind])
            #newspec.listids[ind] = HESS_spec(cname)
            #newspec.listids[ind].threshold
            # Need to fill list of offsets, zenith and efficiencies
            # Need to check thresholds

        return newspec

