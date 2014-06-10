Bounding boxes
==============

.. warning ::

   At the moment this section is pretty much just a set of notes for myself.
   This needs to be cleaned up and translated into code used consistently throughout gammapy
   (and probably astropy and photutils and maybe scikit-image ...)

This document gives a detailed description of bounding boxes.
This is mostly useful for developers writing new functions that use bounding boxes.
Users can consider bounding boxes an implementation detail (even though they are part of the public API in some places).


What's a bounding box and why do we need it?
--------------------------------------------

In gammapy we use `bounding boxes <http://en.wikipedia.org/wiki/Minimum_bounding_box>`_ to speed up image processing.

Let's say you have a large image, but are only interested in a small box (a rectangular sub-image)::

   import numpy as np
   full_array = np.random.random((2000, 3000))
   sub_array = full_array[1000:1010, 2000:2020]
   np.sum(full_array)
   np.sum(sub_array)

Numpy supports working with such boxes efficiently through the concept of 
`slicing <http://scipy-lectures.github.io/intro/numpy/array_object.html#indexing-and-slicing>`__
and 
`views <http://scipy-lectures.github.io/intro/numpy/array_object.html#copies-and-views>`__.

By slicing ``[1000:1010, 2000:2020]`` we created a view (not a copy) ``sub_array`` into the ``full_array``.
On my machine making a measurement on ``sub_array`` is about 1000 times as fast as for ``full_array``::

   In [34]: %timeit np.sum(full_array)
   100 loops, best of 3: 5.32 ms per loop
   
   In [35]: %timeit np.sum(sub_array)
   100000 loops, best of 3: 7.41 µs per loop

How to represent bounding boxes and pass them around in code?
-------------------------------------------------------------

In gammapy we frequently need to bass bounding boxes around, e.g. from a function that detects
many small objects in a large survey image to another function that measures some properties of these objects.  

Python does have a built-in `slice <http://docs.python.org/2/library/functions.html#slice>`__ class,
and we can use it to represent 1-dimensional slices::

   def find_objects(array, threshold):
       """Find segments above threshold in 1-dimensional array."""
       object_slices = []
       in_object = False
       for ii in range(len(array)):
           value = array[ii]
           if (value > threshold) and (not in_object):
               in_object = True
               start = ii
           elif (value <= threshold) and (in_object):
               in_object = False
               stop = ii
               object_slices.append(slice(start, stop))
       return object_slices
   
   def measure_objects(array, object_slices):
       """Measure something for all objects and print the result."""
       for object_slice in object_slices:
           data = array[object_slice]
           measurement = np.sum(data)
           print(object_slice.start, object_slice.stop, measurement)

   import numpy as np
   np.random.seed(42)
   array = np.random.random(1000)
   threshold = 0.99
   objects = find_objects(array, threshold)
   measure_objects(array, objects)


Unfortunately, there is no n-dimensional slice or bounding box class in Python or Numpy.

Bounding boxes in other packages
--------------------------------

Before inventing our own, let's look at what kinds of representations others have come up with:

* The `scipy.ndimage.measurements.find_objects` function returns a Python list of
  Python tuples of Python `slice` objects to represent a list of bounding boxes.
  I.e. a single n-dimensional bounding box is represented as a Python tuple of n slice objects,
  e.g. for a 2-dimensional bounding box::

    bbox = (slice(1, 3, None), slice(2, 5, None))
  
  This has the advantage that for a numpy array it is very easy to create views
  for the rectangles represented by the bounding box::

    array = np.random.random((1000, 2000))
    bboxes = scipy.ndimage.find_objects(array, ...)
    for bbox in bboxes:
        view = array[bbox]
        measurement = np.sum(view)

  As far as I can see no other function except `scipy.ndimage.measurements.find_objects` uses
  this bbox format, though. E.g. all other functions in `scipy.ndimage.measurements` take a
  ``label`` array as input, none has a ``bboxes`` argument.
 
* The `skimage.measure.regionprops` function returns ``properties``, a list of dict-like objects
  with (among many other things) a ``bbox`` entry, which is a Python tuple of integers::

    bbox = (min_row, min_col, max_row, max_col)

  Looking under the hood (this is not part of their API) at the implementation in
  `skimage/measure/_regionprops.py <https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_regionprops.py>`__ ,
  we see that `skimage.measure.regionprops` is just a wrapper storing the `scipy.ndimage.measurements.find_objects` bboxes
  in an object as ``RegionProperties._slice`` and then generating the integer index tuple on demand::

    def bbox(self):
        return (self._slice[0].start, self._slice[1].start,
                self._slice[0].stop, self._slice[1].stop)

  As for `scipy.ndimage`, as far as I can see, ``bbox`` is not used elsewhere in `skimage`. 

* `photutils` has this `coordinate convention <http://photutils.readthedocs.org/en/latest/photutils/index.html#coordinate-convention-in-photutils>`__.
  Looking at the `photutils.aperture_photometry` implementation, it looks like they don't have an official ``bbox`` representation,
  but simply compute ``(x_min, x_max, y_min, y_max)`` where needed and then use ``data[y_min:y_max, x_min:x_max]`` views.
  TODO: update once this is in: https://github.com/astropy/astropy/issues/2607 

* `findobj <http://findobj.readthedocs.org/>`__ doesn't use bounding boxes in the public API.
  Internally they use an ``_ImgCutout`` class and use it in their ``_findobjs`` function.
  
  TODO: briefly describe how it works. 

I also found 
`this <http://stackoverflow.com/questions/9525313/rectangular-bounding-box-around-blobs-in-a-monochrome-image-using-python>`__
and
`this <http://stackoverflow.com/questions/4087919/how-can-i-improve-my-paw-detection>`__
stackoverflow entry a bit useful.


Bounding boxes in gammapy
-------------------------

In gammapy, a single bounding box is represented as a `gammapy.image.measure.BoundingBox` objects.

I decided to make a class, because I think it will help:

* Getting the index order right shouldn't be left up to the user (``(x, y)`` and ``(y, x)`` in different places)
* Getting the position right shouldn't be left up to the user (``(0, 0)`` or ``(1, 1)`` and sub-pixel positions)


TODO: describe. Give examples of functions that take bounding boxes as input or output.

TODO: there should probably also be a ``PixelCoordinate`` class instead of passing ``(x, y)`` tuples around.

