# Licensed under a 3-clause BSD style license - see LICENSE.rst                                
import ctypes

__all__ = ['Image', 'PixelInfo', 'draw_camera']

class Image(ctypes.Structure):
    _fields_ = [
        ("n_pixels", ctypes.c_ushort),
        ("ID", ctypes.POINTER(ctypes.c_ushort)),
        ("intensity", ctypes.POINTER(ctypes.c_float)),
        ("min_intensity", ctypes.c_float),
        ("max_intensity", ctypes.c_float)]

class PixelInfo(ctypes.Structure):
    _fields_ = [
        ("ID", ctypes.c_ushort),
        ("ID_MC", ctypes.c_ushort),
        ("ID_real", ctypes.c_ushort),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("n_nb", ctypes.c_ubyte),
        ("nb", ctypes.POINTER(ctypes.c_ushort))]

def draw_camera():
    #w.create_rectangle(50, 25, 150, 75, fill="blue")
    for i in range(image[image_index].n_pixels):
        print("ID: " + repr(image[image_index].ID[i]) + '    intensity: ' + repr(round(image[image_index].intensity[i], 1)))
    print("")



def init_fits_shower_images(filename):
    """Load images from FITS file."""
    PImage = ctypes.POINTER(Image)
    PPixelInfo = ctypes.POINTER(PixelInfo)

    fitsshowers = CDLL('./_fits_shower_images.so')
    fitsshowers.import_from_fits(filename)
    images = PImage.in_dll(fitsshowers, "im")
    n_images = ctypes.c_int.in_dll(fitsshowers, "n_images").value
    pixel_info = PPixelInfo.in_dll(fitsshowers, "Ppixel_info_HESS2")
    return n_images, images, pixel_info

