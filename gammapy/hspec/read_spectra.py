from sherpa.astro.ui import load_pha,get_data,notice_id,set_source

class HESS_spec():
    """ Class to encapsulate HESS specific data for spectral analysis with sherpa
    """
    def __init__(self,name,filename): # name means datid!!
        self.name = name
        load_pha(name,filename)
        self.data=get_data(name)
        self.arf=get_data(name)
        self.rmf=get_data(name)

        try:         # Read keywords from pha header
            self.threshold = self.data.header['ETH']
        except KeyError:
            print " ! WARNING: no threshold found, using 200 GeV"
            self.threshold = 2e8   # default value 200 GeV
        self.emax = 1e11           # default value 100 TeV

        try:
            self.zenith = self.data.header['ZENITH']
        except KeyError:
            print "WARNING: no mean zenith angle found, using 45 deg"
            self.zenith = 45.0   # default value 200 GeV
 
        try:
            self.offset = self.data.header['OFFSET']
        except KeyError:
            print "WARNING: no offset angle found, using 1.0 deg"
            self.offset = 1.0   # default value 200 GeV

        try:
            self.telcode = self.data.header['TELCODE']
        except KeyError:
            print "WARNING: no telcode found, using 0"
            self.telcode = 0   # default value 200 GeV
            
        #self.exposure = self.data.header['EXPOSURE']

    def set_threshold(self,thres_val):
        self.threshold = thres_val
        
    def set_emax(self,emax):
        self.emax = emax

    def notice(self,emin_ener,emax_ener):
        """ Notice energy range 
            if minimal value required below threshold, use threshold instead
            if maximal value required beyond emax, use emax instead
        """    
        notice_id(self.name,max(self.threshold,min_ener),min(self.emax,max_ener)) 
        
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
    def __init__(self,filelist,name=None):
        self.name = name # source name or tag
        self.list_HESSspecs = []
        
        if filelist is not None:
            self.loadlist(filelist)

    def loadlist(self,listfile):
        """ Load all datasets in listfile
            expect pha files containing keywords for bkg, arf and rmf files
            store dataid
        """        
        for filename in listfile: #assumes file names like run18517.pha
            datid = self.name+filename.split('/')[-1][3:8] 
            self.list_HESSspecs.append(HESS_spec(datid,filename)) 
            # Methods available for HESS_spec objects:
            # set_threshold, set_emax, notice, set_source, set_minimalArea.

    def notice(self,min_ener,max_ener):
        for datid in self.list_HESSspecs:
            datid.notice(min_ener,max_ener)
            
    def set_source(self,model):
        for datid in self.list_HESSspecs:
            datid.set_source(model)
        
    def group(self):
        """ Group spectra 
            first make one big spectrum
            should implement grouping in efficiency/zenith/offset bins
        """ 
        for datid in list_HESSspecs:            # loop over all datasets
            mydat = datid.data
            good_chan=mydat.get_noticed_channels().astype(int)
            totON[good_chan]+=mydat.counts[good_chan]
            totOFF[good_chan]+=mydat.get_background().counts[good_chan]
            tot_alpha[good_chan]+=mydat.get_background().counts[good_chan]*\
                mydat.get_background_scale()
            
            if channel is None:
                channel = mydat.channel
                
            tot_time += mydat.exposure 
            carf = mydat.get_arf().get_y()
            tot_arf += carf*mydat.exposure 
            crmf = mydat.get_rmf()
            chans = crmf.n_chan.cumsum()
                
            tmp_rmf=np.zeros((ntrue,nrec))
            for i in range(ntrue-1):
                tmp_rmf[i+1,crmf.f_chan[i+1]:crmf.f_chan[i+1]+crmf.n_chan[i+1]]=\
                    crmf.matrix[chans[i]:chans[i+1]]*carf[i]                
            tot_rmf += tmp_rmf*mydat.exposure
