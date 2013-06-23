import numpy as np
import logging
logging.basicConfig(level=logging.INFO)



###Blob detect io###

def write_scale_cube(image_cube, scales, filename):
    """Write filtered images to numpy binary file"""
    np.save(filename, image_cube)
    np.save(filename + "_scales", scales)

def read_scale_cube(filename):
    """Load filtered images from numpy binary file"""
    image_cube = np.load(filename)
    scales = np.save(filename + "_scales")
    return scales, image_cube


def write_scale_cube_fits(image, image_cube, scale_parameters, filename, header):
    """Write scale space to fits image cube"""
	from astropy.io import fits

	#Original image as primary HDU
    hdulist = []
    hdulist.append(fits.PrimaryHDU(image))

    #Create image HDU and append it to the list 
    for scale, scale_image in zip(scale_parameters, image_cube) :
        hdu = fits.ImageHDU(data=scale_image, header=header, name="SCALE {0:0.2f}".format(scale))
        hdulist.append(hdu)
    
    #Append scales in a table HDU
    col = fits.Column(name="Scales", format='E', array=scale_parameters)
    cols = fits.ColDefs([col])
    tbhdu = fits.new_table(cols)
    hdulist.append(tbhdu)
    
    #Convert to HDUlist and writeto file    
    HDUlist = fits.HDUList(hdulist)
    HDUlist.writeto(filename)
    logging.info("Writing image cube {0}".format(filename))
    

def read_scale_cube_fits(filename):
    """Read scale space from fits file image cube"""
	from astropy.io import fits

    hdulist = fits.open(filename)
    
    #Get and set up shape
    width, height = hdulist[0].data.shape
    scale_space = np.ndarray(shape = (0, width, height))
    scales = hdulist[-1].data.field("Scales")   
    
    #Read all scale images
    for hdu in hdulist[1:-1]:
        scale_image = np.array(hdu.data, ndmin=3)
        scale_space = np.append(scale_space, scale_image, axis=0)
    
    return scale_space, scales

def write_region_file(regionfile, blobs):
    """Write ds9 region file from blob list"""
    #Open region file, it will be overwritten if it already exists!
    f = open(regionfile, 'w')
    
    #Write blobs to file
    for blob in blobs:
        region_string = "circle({0}, {1}, {2})\n".format(blob.x_pos, blob.y_pos, blob.radius)
        f.write(region_string)
    f.close()
    logging.info("Wrote region file {0}".format(regionfile))
        

###Conversion functions###

def write_fits_table():
    """Write source catalog to fits table"""
	from astropy.table import Table, Column
	catalog = Table()
    
