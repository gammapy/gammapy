import numpy as np
from sherpa.models import ArithmeticModel, Parameter



class Proton(ArithmeticModel):
    """
    Model definition for a proton power law with exponential cutoff
    Author: regis.terrier@apc.univ-paris7.fr
    """

    def __init__(self, name='proton'):
        # First precompute some quantities 
        self.Ep_min = 1e-1  # TeV
        self.Ep_max = 1e5   # TeV
        self.nbins = 300
        self.lEp_min = np.log10(self.Ep_min)
        self.lEp_max = np.log10(self.Ep_max)
        self.Ep = np.logspace(self.lEp_min,self.lEp_max,self.nbins)
        self.lbsize = (self.lEp_max-self.Ep_min)/self.nbins
        self.Fgam = None
        self.EG = None
        self.EP = None
        self.ncalc=0

        # Instantiate parameters
        self.Eo = Parameter(name, 'Eo', 10, frozen=True, units='TeV') # p[0] Normalized at 10 TeV by default
        self.beta = Parameter(name, 'beta', 1., min=1e-3, max=1e4, units='1/PeV')        #p[1]
        self.gamma = Parameter(name, 'gamma', 2.2, min=-1, max=5)                           #p[2]
        self.ampl = Parameter(name, 'ampl', 1e-11, min=1e-15, max=1e15, units='1/cm^2/s/TeV') #p[3]
        self.Einf = Parameter(name, 'Einf', 1, frozen=True, units='TeV') # p[4] 1 TeV by default
        self.Esup = Parameter(name, 'Esup', 100, frozen=True, units='TeV') # p[5] 100 TeV by default

                
        ArithmeticModel.__init__(self, name, (self.Eo,self.beta,self.gamma,self.ampl,self.Einf,self.Esup))
        

    def F_gamma(self,EG,EP):
        """
        E_gamma:           photon energy in TeV
        """
        x=EG/EP
        valid = (x<1.0)
        L=np.log(EP)
        L2 = L**2 
        p=1/(1.79+0.11*L+0.008*L2)
        B_gam=1.3+0.14*L+0.011*L2
        k_gam=1/(0.801+0.049*L+0.014*L2)
        
        sigma=(34.3+1.88*L+0.25*L2)*1e-27
        logx = np.log(x)
        
        xp = x**p
        mxp = 1-xp
        tmp = 1+k_gam*xp*mxp
        
        res1=B_gam*logx/x*(mxp/tmp)**4
        res2=1./logx - 4*p*xp/mxp -4*k_gam*p*xp*(1-2*xp)/tmp
        return res1*res2*sigma*valid

    def pflux(self,p,x):        
        Egam = x*1e-9 # in TeV
        
        if self.EG is None:
            self.EG,self.EP = np.meshgrid(Egam,self.Ep)
            self.Fgam = self.F_gamma(self.EG,self.EP)
        elif np.array_equal(Egam,self.EG[0,:]) is False:
            print Egam.shape,self.EG.shape
            print "Warning different internal vectors. Recomputing."
            self.EG,self.EP = np.meshgrid(Egam,self.Ep)
            self.Fgam = self.F_gamma(self.EG,self.EP)

#        proton_spec = p[3]*np.exp(-self.EP*p[1]*1e-3)*(self.EP/p[0])**(-p[2])
        proton_spec = np.exp(-self.EP*p[1]*1e-3)*(self.EP/p[0])**(-p[2])
        
        if self.Fgam is None:
            self.Fgam = self.F_gamma(self.EG,self.EP)

        res = self.Fgam*proton_spec
        integral =  3e10*res.sum(0)*self.lbsize/np.log(10.)

        # Normalize with proton energy content at 1 kpc for n_H = 1 cm^-3
        pe_spec = self.Ep**2 * np.exp(-self.Ep*p[1]*1e-3) * (self.Ep/p[0])**(-p[2])
        pe_spec =pe_spec[(self.Ep>=p[4])*(self.Ep<=p[5])]
        norm = 4*np.pi * (3.1e21)**2 * 1.602e9 * pe_spec.sum()*self.lbsize/np.log(10.)/1e50
        
#        return 3e10*res.sum(0)*self.lbsize/np.log(10.)*1e-9   # per TeV not keV
        return  integral/norm*p[3]


    def point(self, p, x):
        """
         point version, 
         
        Params
        `p`  list of ordered parameter values.
        `x`  ndarray of bin midpoints.
        
        returns ndarray of calculated function values at
                bin midpoints
        """
        return self.pflux(p,x)

    def integrated(self, p, xlo, xhi):
        """        
         integrated form from lower bin edge to upper edge 
        
        Params
        `p`   list of ordered parameter values.
        `xlo` ndarray of lower bin edges.
        `xhi` ndarray of upper bin edges.

        returns ndarray of integrated function values over 
                lower and upper bin edges.
        """
#        print "calc",p
        val1 = np.zeros(xhi.shape[0]+1)
        val1[1:] = self.pflux(p,xhi)
        val1[0] =  val1[1]#self.pflux(p,xlo[0]) 
        flux = (xhi-xlo)*(val1[:-1]+val1[1:])*0.5
#        flux = (xhi-xlo)*(self.pflux(p,xhi)+self.pflux(p,xlo))*0.5 # Trapezoid integration
        return flux   

    def calc(self, p, xlo, xhi=None, *args, **kwargs):        
        """
        Params
        `p`   list of ordered parameter values.
        `x`   ndarray of domain values, bin midpoints or lower
              bin edges.
        `xhi` ndarray of upper bin edges.

        returns ndarray of calculated function values.
        """
        self.ncalc += 1
        if self.ncalc % 1000 == 0:
            print self.ncalc, p

        if xhi is None:
            return self.point(p, xlo)
        return self.integrated(p, xlo, xhi)     

        
