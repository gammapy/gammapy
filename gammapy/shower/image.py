# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from ..shower import hillas_parameters

__all__ = ['CherenkovCamera', 'ShowerImage', 'ShowerImages']


def clean_image_double_threshold(pix_id, neighbors, value, value_min_1, value_min_2):
    """Clean an image (i.e. select a subset of pixels based on some criterion)"""
    raise NotImplementedError

class CherenkovCamera(object):
    """Cherenkov camera layout.
    
    Parameters
    ----------
    pix_id : array-like
        Pixel identification number (ID)
    x : array-like
        Pixel position x-coordinates
    y : array-like
        Pixel position y-coordinates
    neighbors : 2D array-like
        neighbors[i][j] is the ID of neighbor j for pixel i

    TODO: Should we use list or dict for neighbors instead?
    """
    
    def __init__(self, pix_id, x, y, neighbors):
        self.pix_id = pix_id
        self.x = x
        self.y = y
        self.neighbors = neighbors
        
    def read(self, filename):
        raise NotImplementedError
    
    def write(self, filename):
        raise NotImplementedError



class ShowerImage(object):
    """Air shower image.

    TODO: implement.
    - neighbor list iterator
    - I/O
    - unit test
    - plotting
    - Gauss model fit
    """
    def __init__(self, camera, pix_id, value):
        self.camera = camera
        self.pix_id = pix_id
        self.value = value
    
    def hillas_parameters(self):
        """Compute Hillas parameters.
        """
        x = self.camera.get_x(self.pix_id)
        y = self.camera.get_y(self.pix_id)
        value = self.value
        return hillas_parameters(x, y, value)
    
    def clean_image(self, method='double_threshold', parameters=dict()):
        pix_id = self.pix_id
        neighbors = self.camera.get_neighbors(self.pix_id)
        value = self.value
        
        if method == 'double_threshold':
            new_pix_id, new_value = clean_image_double_threshold(pix_id, neighbors, value, **parameters)
        else:
            s = 'Unknown method: {0}.\n'.format(method)
            s += 'Available methods: double_threshold.'
            raise ValueError(s)
        new_shower_image = ShowerImage(self.camera, new_pix_id, new_value)
        return new_shower_image


class ShowerImages(object):
    """A list of ShowerImage objects"""

    def __init__(self, camera):
        self.camera = camera
    
    def read(self, filename):
        pass
    
    def write(self, filename):
        pass
    
    def next(self):
        pass
        # TODO: define iterator
    
    def hillas_parameters(self):
        """Returns a table of Hillas parameters"""
        pass
    
    def clean_images(self):
        """Returns a new ShowerImages object where each ShowerImage has been cleaned"""
        pass


"""
TODO: An alternative would be to use functions instead of making everythin class methods. Is that better?

def clean_image(camera, shower):
    raise NotImplementedError

def hillas_parameters():
    pass

"""

if __name__ == '_main__':
    camera = CherenkovCamera.read('camera.fits')
    shower_images = ShowerImages(camera).read('showers.fits')
    for shower_image in shower_images:
        print shower_image.hillas_parameters()
