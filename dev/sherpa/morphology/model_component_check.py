"""
Illustration that get_model_component_image doesn't apply PSF convolution in this case,
even though according to the docs it should:
http://cxc.harvard.edu/sherpa/ahelp/get_model_component_image.html

Sherpa bug? 
"""
# Set up dataspace
dataspace2d((101, 101))

# Set source 
set_source(normgauss2d.gauss)
gauss.ypos = 50
gauss.xpos = 50
gauss.fwhm = 5

# Load PSF
load_psf("psf", normgauss2d.gauss_psf)
set_psf(psf)

# Save source and model
save_model("model.fits")
save_source("source.fits")

# Save source and model components
from crates_contrib.utils import make_image_crate

component_model = np.array(get_model_component_image("gauss").y, dtype=np.float32)
crate_model = make_image_crate(component_model)
crate_model.write("model_component.fits")

component_source = np.array(get_source_component_image("gauss").y, dtype=np.float32)
crate_source = make_image_crate(component_source)
crate_source.write("source_component.fits")
