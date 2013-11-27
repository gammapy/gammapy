"""Some test for the normdisk2d and normshell2d models.

Everything is OK:
- The Sherpa models integrate to 1
- The Sherpa models give consistent results with gammalib
"""

def check_integrals():
    """Check that Sherpa normed models integrate to 1."""
    from sherpa.astro import ui
    from sherpa.astro.ui import normgauss2d
    from models import normdisk2d, normshell2d
    
    ui.clean()
    
    g = normgauss2d('g')
    g.xpos, g.ypos, g.ampl, g.fwhm = 100, 100, 42, 5
    
    d = normdisk2d('d')
    d.xpos, d.ypos, d.ampl, d.r0 = 100, 100, 42, 50
    
    s = normshell2d('s')
    s.xpos, s.ypos, s.ampl, s.r0, s.width = 100, 100, 42, 30, 20
    
    models = [g, d, s]
    
    ui.dataspace2d((200, 200))
    for model in models:
        ui.set_model(model)
        # In sherpa normed model values are flux per pixel area.
        # So to get the total flux (represented by the `ampl` parameter)
        # one can simply sum over all pixels, because a pixel has area 1 pix^2.
        # :-) 
        integral = ui.get_model_image().y.sum()
        print model.name, integral
"""Output of check_integrals:
normgauss2d.g 42.0
d 41.9519697595
s 41.9941121559
""" 

# Example parameters used in the following two functions (in pix, as used by Sherpa)
INTEGRAL = 42
GAUSS_FWHM = 5
DISK_R0 = 7
SHELL_R0 = 5
SHELL_WIDTH = 5

THETAS = [0, 3, 5, 7, 10]

PIX_TO_DEG = 0.01 # arbitrary


def print_values_sherpa():
    """Print some Sherpa model values that can be used for unit tests."""
    from sherpa.astro import ui
    from sherpa.astro.ui import normgauss2d
    from models import normdisk2d, normshell2d
    
    ui.clean()
    
    g = normgauss2d('g2')
    g.ampl, g.fwhm = INTEGRAL, GAUSS_FWHM
    
    d = normdisk2d('d')
    d.ampl, d.r0 = INTEGRAL, DISK_R0
    
    s = normshell2d('s')
    s.ampl, s.r0, s.width = INTEGRAL, SHELL_R0, SHELL_WIDTH

    models = [g, d, s]
    for model in models:
        for theta in THETAS:
            value = model(0, theta)
            print model.name, theta, value
"""Output of print_values_sherpa:
normgauss2d.g2 0 1.48267123303
normgauss2d.g2 3 0.546464139553
normgauss2d.g2 5 0.0926669520641
normgauss2d.g2 7 0.0064709746041
normgauss2d.g2 10 2.26237675938e-05
d 0 0.2728370453
d 3 0.2728370453
d 5 0.2728370453
d 7 0.2728370453
d 10 0.0
s 0 0.114591559026
s 3 0.126953513392
s 5 0.198478402352
s 7 0.16366948346
s 10 0.0
"""

def print_values_gammalib():
    """Print some Gammalib model values that can be used for unit tests.
    
    Gammalib uses normalised PDFs, i.e. eval() returns probability / steradian
    so that the integral is one.
    """
    import numpy as np
    import gammalib
    
    # We need some dummy variables
    center = gammalib.GSkyDir()
    center.radec_deg(0, 0)
    energy = gammalib.GEnergy()
    time   = gammalib.GTime()

    FWHM_TO_SIGMA = 1. / np.sqrt(8 * np.log(2))
    g = gammalib.GModelSpatialRadialGauss(center, PIX_TO_DEG * FWHM_TO_SIGMA * GAUSS_FWHM)
    d = gammalib.GModelSpatialRadialDisk(center, PIX_TO_DEG * DISK_R0)
    s = gammalib.GModelSpatialRadialShell(center, PIX_TO_DEG * SHELL_R0, PIX_TO_DEG * SHELL_WIDTH)

    models = [('g', g), ('d', d), ('s', s)]
    for name, model in models:   
        for theta in THETAS:
            theta_radians = np.radians(PIX_TO_DEG * theta)
            gammalib_value = model.eval(theta_radians, energy, time)
            sherpa_value = INTEGRAL * gammalib_value * np.radians(PIX_TO_DEG) ** 2
            print name, theta, sherpa_value
"""Output of print_values_gammalib:
g 0 1.48267123303
g 3 0.546464139553
g 5 0.0926669520641
g 7 0.0064709746041
g 10 2.26237675938e-05
d 0 0.272837079247
d 3 0.272837079247
d 5 0.272837079247
d 7 0.272837079247
d 10 0.0
s 0 0.114591559026
s 3 0.126953513392
s 5 0.198478402352
s 7 0.16366948346
s 10 0.0
"""

if __name__ == '__main__':
    #check_integrals()
    #print_values_sherpa()
    print_values_gammalib()
