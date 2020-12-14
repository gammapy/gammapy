.. include:: ../references.txt

.. _regionmap:

******************************
The RegionGeom and RegionNDMap
******************************

.. currentmodule:: gammapy.maps

This page provides examples and documentation specific to the Region
classes. 

RegionGeom
==========
A `RegionGeom` is analogous to a  map geometry `Geom`, but instead of a fine grid on a rectangular region, 
it is made up of a single large pixel with an arbitrary shape that can also have any 
number of non-spatial dimensions.

Creating a RegionGeom
---------------------

.. code-block:: python

    from astropy.coordinates import SkyCoord
    from regions import CircleSkyRegion, RectangleSkyRegion
    from gammapy.maps import RegionGeom
    import astropy.units as u

    # Create a circular region with radius 1 deg centered around
    # the Galactic Center
    center = SkyCoord("0 deg", "0 deg", frame="galactic")
    region =  CircleSkyRegion(center=center, radius=1*u.deg)
    geom = RegionGeom(region)

    # Create a rectangular region with a 45 degree tilt
    region =  RectangleSkyRegion(center=center, width=1*u.deg, height=2*u.deg, angle=45*u.deg)
    geom = RegionGeom(region)

    # Equivalent factory method call
    geom = RegionGeom.create(region) 


Higher dimensional region geometries (cubes and hypercubes) can be constructed by passing a list of `MapAxis` objects for non-spatial dimensions with the axes parameter:

.. code-block:: python

    from astropy.coordinates import SkyCoord
    from regions import CircleSkyRegion
    from gammapy.maps import MapAxis, RegionGeom
    import astropy.units as u
    center = SkyCoord("0 deg", "0 deg", frame="galactic")
    region =  CircleSkyRegion(center=center, radius=1*u.deg)
    energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
    geom = RegionGeom(region, axes=[energy_axis])

The resulting `RegionGeom` object has `ndim = 3`, two spatial dimensions with one single bin and the chosen energy axis with 12 bins:

.. code-block:: python

    >>> geom

    RegionGeom
            region     : CircleSkyRegion
            axes       : ['lon', 'lat', 'energy']
            shape      : (1, 1, 12)
            ndim       : 3
            frame      : galactic
            center     : 0.0 deg, 0.0 deg

RegionGeom and coordinates
--------------------------
A `RegionGeom` defines a single spatial bin with arbitrary shape. The spatial coordinates are then given by the center of the region geometry. If one or more non-spatial axis are present, 
they can have any number of bins. There are different methods that can be used to access or modify the coordinates of a `RegionGeom`.

+ Bin volume and angular size:
    The angular size of the region geometry is given by the method `RegionGeom.solid_angle()`. If a region geometry has any number of non-spatial axes, then the volume of each bin is given by `RegionGeom.bin_volume()`.
    If there are no non-spatial axes, both return the same quantity.

    .. code-block:: python

        from astropy.coordinates import SkyCoord
        from regions import CircleSkyRegion
        from gammapy.maps import MapAxis, RegionGeom
        import astropy.units as u

        # Create a circular region geometry with an energy axis
        center = SkyCoord("0 deg", "0 deg", frame="galactic")
        region =  CircleSkyRegion(center=center, radius=1*u.deg)
        energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
        geom = RegionGeom(region, axes=[energy_axis])

        # Get angular size and bin volume
        angular_size = geom.solid_angle()
        bin_volume = geom.bin_volume()
    
+ Coordinates defined by the `RegionGeom`:
    Given a map coordinate or `MapCoord` object, the method `RegionGeom.contains(coords)` checks if they are contained in the region geometry. One can also retrieve the coordinates of the region geometry with
    `RegionGeom.get_coord()` and `RegionGeom.get_idx()`, which return the sky coordinates and indexes respectively. Note that the spatial coordinate will always be a single entry, namely the center, while any non-spatial 
    axis can have as many bins as desired.

    .. code-block:: python

        from astropy.coordinates import SkyCoord
        from regions import CircleSkyRegion
        from gammapy.maps import RegionGeom
        import astropy.units as u

        # Create a circular region geometry
        center = SkyCoord("0 deg", "0 deg", frame="galactic")
        region =  CircleSkyRegion(center=center, radius=1*u.deg)
        geom = RegionGeom(region)

        # Get coordinates and indexes defined by the region geometry
        coordinates = geom.get_coord()
        indexes = geom.get_idx()

        # Check if a coordinate is contained in the region
        >>> geom.contains(center)
            True


+ Upsampling and downsampling the non-spatial axes:
    The spatial binning of a `RegionGeom` is made up of a single bin, that cannot be modified as it defines the region. However, if any non-spatial axes are present, they can be modified using the 
    `RegionGeom.upsample()` and `RegionGeom.downsample()` methods, which take as input a factor by which the indicated axis is to be up- or downsampled.

    .. code-block:: python

        from astropy.coordinates import SkyCoord
        from regions import CircleSkyRegion
        from gammapy.maps import MapAxis, RegionGeom
        import astropy.units as u

        # Create a circular region geometry with an energy axis
        center = SkyCoord("0 deg", "0 deg", frame="galactic")
        region =  CircleSkyRegion(center=center, radius=1*u.deg)
        energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
        geom = RegionGeom(region, axes=[energy_axis])

        # Upsample the energy axis by a factor 2
        geom_24_energy_bins = geom.upsample(2, "energy")
        # Downsample the energy axis by a factor 2
        geom_6_energy_bins = geom.downsample(2, "energy")

+ Image and WCS geometries:
    * If a `RegionGeom` has any number of non-spatial axis, the corresponding region geometry with just the spatial dimensions is given by the method `RegionGeom.to_image()`. If the region geometry only has spatial
      dimensions, a copy of it is returned.
    * Conversely, non-spatial axis can be added to an existing `RegionGeom` by `RegionGeom.to_cube()`, which takes a list of non-spatial axes with unique names to add to the region geometry.    
    * Region geometries are made of a single spatial bin, but are constructed on top of a finer `WcsGeom`. The method `RegionGeom.to_wcs_geom()` returns the minimal equivalent geometry that contains the region geometry.
      It can also be given as an argument a minimal width for the resulting geometry.

    .. code-block:: python

        from astropy.coordinates import SkyCoord
        from regions import CircleSkyRegion
        from gammapy.maps import MapAxis, RegionGeom
        import astropy.units as u

        # Create a circular region geometry
        center = SkyCoord("0 deg", "0 deg", frame="galactic")
        region =  CircleSkyRegion(center=center, radius=1*u.deg)
        geom = RegionGeom(region)

        # Add an energy axis to the region geometry
        energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
        geom_energy = geom.to_cube[energy_axis]

        # Get the image region geometry without energy axis. 
        # Note that geom_image == geom
        geom_image = geom_energy.to_image()

        # Get the minimal wcs geometry that contains the region
        wcs_geom = geom.to_wcs_geom()

RegionNDMap
===========
A `RegionNDMap` owns a `RegionGeom` instance as well as a data array containing map values.
It can be thought of as a `Map` but with a single spatial bin that can have an arbitrary 
shape, together with any non-spatial axis.
It is to a `RegionGeom` what a `Map` is to a `Geom`.