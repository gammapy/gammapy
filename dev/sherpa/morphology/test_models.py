"""Some test for the normdisk2d and normshell2d models"""

def check_integrals():
    """Check that Sherpa normed models integrate to 1."""
    pass

def print_values_sherpa():
    """Print some Sherpa model values that can be used for unit tests."""
    from sherpa.astro.ui import normgauss2d
    from models import normdisk2d, normshell2d
    
    g = normgauss2d('g')
    g.ampl, g.fwhm = 42, 5
    
    d = normdisk2d('d')
    d.ampl, d.r0 = 42, 7
    
    s = normshell2d('s')
    s.ampl, s.r0, s.width = 42, 5, 5

    models = [g, d, s]
    thetas = [0, 3, 5, 7, 10]    
    for model in models:
        for theta in thetas:
            print model.name, theta, model(0, theta)

def print_values_gammalib():
    """Print some Gammalib model values that can be used for unit tests."""
    pass


if __name__ == '__main__':
    #check_integrals()
    print_values_sherpa()
    #print_values_gammalib()
