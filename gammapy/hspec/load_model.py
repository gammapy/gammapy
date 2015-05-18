from sherpa.astro.ui import *

def model_io():
    while 1:
        model = raw_input(" - Please choose a model or a combination of them (Enter ? to see options) ")
        if model == '?':
            show_models()
        else:
            break
    return model

def show_models():    
    ''' Prints a list of available models
    '''
    print " * Sherpa models:"
    print "  ", list_models()
    print " * Own models:"
    print "   plexpcutoff, Finke, Franceschini"
    print " * GammaFit models:"
    print "   synchro, ic, pion"

def prod(components):
    ''' Performs product of multiplicative model components
    '''
    fact = 1
    for comp in components:
        fact *= comp
    return fact

def load_model(modelstring):
    ''' Performs the arithmetic combination of the model components
    '''
    model_terms = modelstring.rsplit('+')    
    components = []

    for term in model_terms:
        model_facts = term.rsplit('*')    
        nfacts  = len(model_facts)
        factors = [ assign_model(model_facts[i],i) for i in range(nfacts)]
        components.append(prod(factors))  
        
    totmodel = sum(components)            
    components.append(totmodel)
    return components 
    #return totmodel

def assign_model(model_name,i):
    ''' Dedicated set up for the most common models.
    '''
    if model_name == 'powlaw1d':
        from sherpa.models import PowLaw1D 
        p1 = PowLaw1D('PowLaw'+str(i))
        p1.gamma = 2.6
        p1.ampl  = 1e-20
        p1.ref   = 1e9
        freeze(p1.ref)        
    elif model_name == 'logparabola':
        p1 = logparabola('LogPar'+str(i))
        p1.ampl = 1e-20
        p1.c1   = 2.
        p1.c2   = 1.
        p1.ref  = 1e9
        freeze(p1.ref)
    elif model_name == 'plexpcutoff': # all parameters in TeV here
        from models.plexpcutoff import *
        p1 = MyPLExpCutoff('PLexpCutoff'+str(i))
        p1.gamma = 2.
        p1.No    = 1e-11
        p1.beta  = 1e-1 # 1/Ecutoff
        p1.Eo    = 1
        freeze(p1.Eo)
    elif model_name == 'Finke': # EBL model from Finke et al. 2010
        enable_table_model()
        load_table_model('p1','/home/ignasi/sw/Hspec/models/frd_abs.fits')
    elif model_name == 'Franceschini': # EBL model from Franceschini et al. 2012
        enable_table_model()
        load_table_model('p1','/home/ignasi/sw/Hspec/models/ebl_franceschini.fits')        
    elif model_name == 'synchro':
        #print 'Synchrotron model not available yet, sorry.'
        #quit() # Stops with an error: ValueError: slice step cannot be zero    
        from gammafit.sherpamod import Synchrotron
        p1 = Synchrotron('Synchro'+str(i))        
        p1.index = 2.
        p1.ampl  = 1.
    elif model_name == 'ic': # Inverse Compton peak
        # Weird, it returns the fit results twice (actually, it seems to do everything twice) 
        # Returns error except if --noplot: TypeError: calc() takes exactly 4 arguments (3 given)
        from gammafit.sherpamod import InverseCompton
        p1 = InverseCompton('IC'+str(i))
        p1.index = 2.
        p1.ampl  = 1e-7 # Not sure if the units are OK
    elif model_name == 'pion': # Pion-decay gamma-ray spectrum
        # Veeery slow convergence
        # also doubled operations and problems for plotting, like in ic.
        from gammafit.sherpamod import PionDecay
        p1 = PionDecay('Pion'+str(i))        
        p1.index = 2.
        p1.ampl  = 10.
    else: # set initial parameter values manually
          # (user-defined models and models that need some extra import will not work)
        p1 = globals()[model_name](model_name+str(i))
        set_manual_model(p1)         
    return p1

def set_manual_model(model,par_array=None):
    ''' Sets initial values for model parameters one by one or as an array
    '''
    print model
    for i in range(len(model.pars)):
        para = model.pars[i]        
        if par_array==None: # one by one
            while True:
                print para.fullname, '=',  para.val
                new_val = raw_input("New parameter value? (Enter to leave unchanged)\n")
                if new_val == '':
                    break
                new_val = float(new_val)
                if para.min < new_val < para.max:
                    para.val = new_val
                    while True:
                        freeze = raw_input("Freeze parameter? (y/n, default = NO)\n")
                        freeze = freeze.lower()
                        if freeze == 'y':
                            para.freeze()
                        elif freeze == 'n' or freeze == '':
                            para.thaw()
                        else:
                            continue
                        break
                    else: 
                        print "Attempted to set parameter out of bounds"
                        continue
                    break
        else: # all parameters at once
            para.val = par_array[i]

def enable_table_model():  # Absolutely not tested
    try: 
        from sherpa.astro.xspec.XSTableModel import *            
        #from sherpa.utils import linear_interp
    except ImportError:
        print ' ! Xspec models are not available, sorry'
        quit()
