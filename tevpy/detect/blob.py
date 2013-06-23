"""Blob detection tool for source detection

This generic blob detection algorithm is based on:
http://www.cs.utah.edu/~jfishbau/advimproc/project1/ (04.04.2013)
Theory behind: http://en.wikipedia.org/wiki/Blob_detection (04.04.2013)
I modified the peak detection by using a maximum filter.
"""
import logging
logging.basicConfig(level=logging.INFO)
from itertools import combinations
import numpy as np
from numpy import sqrt, sin, cos, pi, arccos, abs, exp

from scipy.ndimage import gaussian_filter, gaussian_laplace, maximum_filter
from scipy.ndimage.morphology import binary_erosion

__all__ = ['create_scale_space', 'detect_peaks', 'detect_peaks_3D',
           'show_peaks', 'detect_blobs_3D', 'detect_blobs', 'prune_blobs',
           'show_blobs', 'Blob']


def create_scale_space(image, scales, filter_='gaussian_laplace'):
    """Creates Scale Space for a given image and stores it in 3D array"""
    # Filter option
    if filter_ == 'gaussian':
        scale_filter = gaussian_filter
        N = '1' 
    elif filter_ == 'gaussian_laplace':
        scale_filter = gaussian_laplace
        N = '-scale**2'  # Normalization for linear scale space, see Wikipedia link above
    else:
        logging.error('Invalid filter option')
    
    # Set up scale space dimensions
    width, height = image.shape
    scale_space = np.ndarray(shape=(0, width, height))
    
    # Compute scale space
    for scale in scales:
        logging.info("Computing scale {0}:".format(scale))
        image_scaled = eval(N) * np.array(scale_filter(image, scale, mode='constant'), ndmin=3)
        scale_space = np.append(scale_space, image_scaled, axis=0)
    return scale_space        
 

def detect_peaks(image):
    """Detect peaks in an image  using a maximum filter"""
    # Set up 3x3 footprint for maximum filter, can also be a different neighborhood if necessary
    footprint = np.ones((3, 3))

    # Apply maximum filter: All pixel in the neighborhood are set
    # to the maximal value. Peaks are where image = maximum_filter(image)
    local_maximum = maximum_filter(image, footprint=footprint) == image
    
    # We have false detections, where the image is zero, we call it background.
    # Create the mask of the background
    zero_background = (image == 0)

    # Erode background at the borders, otherwise we would miss the points in the neighborhood 
    eroded_background = binary_erosion(zero_background, structure=footprint, border_value=1)

    # Remove the background from the local_maximum image
    detected_peaks = local_maximum - eroded_background
    return detected_peaks


def detect_peaks_3D(image):
    """Same functionality as detect_peaks, but works on image cubes. """
    # Set up 3x3 footprint for maximum filter
    footprint = np.ones((3, 3, 3))

    # Apply maximum filter: All pixel in the neighborhood are set
    # to the maximal value. Peaks are where image = maximum_filter(image)
    local_max = maximum_filter(image, footprint=footprint, mode='constant') == image
    
    # We have false detections, where the image is zero, we call it background.
    # Create the mask of the background
    background = (image == 0)

    # Erode background at the borders, otherwise we would miss 
    eroded_background = binary_erosion(background, structure=footprint, border_value=1)

    # Remove the background from the local_max mask
    detected_peaks = local_max - eroded_background
    return detected_peaks
    
 
def show_peaks(image_3D):
    """Show all images of different scales including the detected peaks. Useful for debugging.""" 
    import matplotlib.pyplot as plt

    for scale_image in image_3D:
        # Detect peaks
        detected_peaks = detect_peaks(scale_image)
        
        # Show image and peaks
        plt.imshow(scale_image)
        x, y = np.where(detected_peaks == 1)
        plt.scatter(y, x)
        plt.show()
    
 
def detect_blobs_3D(image, threshold):
    """Find maxima in image cubes"""
    # Replace nan values by 0
    image = np.nan_to_num(image)
    
    # Compute scale parameters and compute scale space   
    scale_parameters = np.linspace(1, 30, 50)
    image_3D = create_scale_space(image, scale_parameters)
    blobs = []
    
    # Employ threshold
    mask_threshold = image_3D > threshold
    detected_peaks = detect_peaks_3D(image_3D * mask_threshold)
    scale_list, y_list, x_list = np.where(detected_peaks == 1)
    
    # Loop over all found blobs     
    for x, y, scale in zip(x_list, y_list, scale_list):
        val = image_3D[scale][y][x]
        blobs.append(Blob(x, y, scale, val))
    logging.info('Found {0} blobs.'.format(len(blobs)))    
    return blobs
    
 
def detect_blobs(image_3D, scales, threshold):
    """Detect blobs of different sizes"""
    # Set up empty blob list
    blobs = []
    
    # Loop over all scale space images 
    for i, scale_image in enumerate(image_3D):
        # Maybe it is useful to employ different threshold values on different scales
        mask_threshold = scale_image > threshold  
        
        # Detect peaks
        detected_peaks = detect_peaks(scale_image * mask_threshold)
        
        # Get peak coordinates
        y_list, x_list = np.where(detected_peaks == 1)
        
        # Set up list of blobs, with position, norm and size
        for x, y in zip(x_list, y_list):
            scale = scales[i]
            value = scale_image[y][x]
            blobs.append(Blob(x, y, scale, value))
    return blobs
        
 
def prune_blobs(blobs, overlap_threshold, q_factor):
    """Prune blobs. If the overlap area of two blobs is to large, the one with the smaller peak value is dismissed"""
    # It is still the question whether the result is unique
    # Loop over all pairwise blob combinations
    for blob_1, blob_2 in combinations(blobs, 2):
        if q_factor:
            overlap = blob_1.q_factor(blob_2)
        else:    
            overlap = blob_1.overlap(blob_2)
        if overlap > overlap_threshold:  # Overlap criterion, neighborhood criterion
                if blob_1.value > blob_2.value:  # Find maximum
                    blob_2.keep = False
                else:
                    blob_1.keep = False
    return [blob for blob in blobs if blob.keep]  # That is Python programming at its best:-)


def show_blobs(image, blobs):
    """Show input image with overlaid blobs"""
    import matplotlib.pyplot as plt

    plt.imshow(image, origin='lower')
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    # plt.colorbar()
       
    for blob in blobs:
        logging.info('Found blob: {0}'.format(blob))
        x, y = blob.image() 
        plt.plot(x, y, color='y')
    plt.show()


class Blob(object):
    """An excess blob is represented by a position, radius and peak value."""  
    def __init__(self, x_pos, y_pos, radius, value):
        self.x_pos = x_pos
        self.y_pos = y_pos
        # The algorithm is most sensitive for extensions of sqrt(2) * t, where t is the scale space parameter
        # This has still to be verified, e.g. for a Gaussian source
        self.radius = 1.41 * radius 
        self.value = value
        self.keep = True
        
        
    def area(self):
        """Compute blob area"""
        return pi * self.radius ** 2
    
    
    def overlap(self, blob):
        """Compute overlap between two blobs. Defined by the overlap area."""
        # For now it is just the overlap area of two containment circles
        # It could be replaced by the Q or C factor, which also defines
        # a certain neighborhood. 
         
        d = sqrt((self.x_pos - blob.x_pos) ** 2 + (self.y_pos - blob.y_pos) ** 2) 
        
        # One circle lies inside the other
        if d < abs(self.radius - blob.radius):
            area = pi * min(self.radius, blob.radius) ** 2
        
        # Circles don't overlap    
        elif d > (self.radius + blob.radius):
            area = 0
            
        # Compute overlap area. Reference: http://mathworld.wolfram.com/Circle-CircleIntersection.html (04.04.2013)
        else:   
            area = (blob.radius ** 2 * arccos((d ** 2 + blob.radius ** 2 - self.radius ** 2) / (2 * d * blob.radius)) 
                                + self.radius ** 2 * arccos((d ** 2 + self.radius ** 2 - blob.radius ** 2) / (2 * d * self.radius)) 
                                - 0.5 * sqrt(abs((-d + self.radius + blob.radius) * (d + self.radius - blob.radius) * 
                                                 (d - self.radius + blob.radius) * (d + self.radius + blob.radius))))
        return max(area / self.area(), area / blob.area())
    
    
    def q_factor(self, blob):
        """Compute q factor as overlap criterion"""
        # Approx HESS PSF size
        sigma_PSF = 0.1
        
        # Compute convolved sigma 
        sigma_A = sqrt(self.radius ** 2 + sigma_PSF ** 2)
        sigma_B = sqrt(blob.radius ** 2 + sigma_PSF ** 2) 
        
        # sigma_AB squared
        sigma_AB2 = sigma_A ** 2 + sigma_B ** 2 
        
        # displacement x_AB squared  
        x_AB2 = (self.x_pos - blob.x_pos) ** 2 + (self.y_pos - blob.y_pos) ** 2 
        
        # Normalization constant
        N = 2. * sigma_A * sigma_B / sigma_AB2 
        return N * exp(-0.5 * x_AB2 / sigma_AB2)
        

    def image(self):
        """Return image of the blob"""
        phi = np.linspace(0, 2 * pi, 360)
        x = self.radius * cos(phi) + self.x_pos
        y = self.radius * sin(phi) + self.y_pos 
        return x, y


    def __str__(self):
        """Is called by the print statement"""
        fmt = 'x_pos: {0}, y_pos: {1}, radius: {2:02.2f}, peak value: {3:02.2f}'
        return fmt.format(self.x_pos, self.y_pos, self.radius, self.value)

