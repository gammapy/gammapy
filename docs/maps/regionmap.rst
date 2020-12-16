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
A `~RegionGeom` is analogous to a  map geometry `~Geom`, but instead of a fine grid on a rectangular region, 
it is made up of a single large pixel with an arbitrary shape that can also have any 
number of non-spatial dimensions.

Creating a RegionGeom
---------------------
A `~RegionGeom` can be created via a DS9 region string (see http://ds9.si.edu/doc/ref/region.html for a list of options)
or an Astropy Region (https://astropy-regions.readthedocs.io/en/latest/).

.. code-block:: python

    from astropy.coordinates import SkyCoord
    from regions import CircleSkyRegion, RectangleSkyRegion
    from gammapy.maps import RegionGeom
    import astropy.units as u

    # Create a circular region with radius 1 deg centered around
    # the Galactic Center
    # from DS9 string
    geom = RegionGeom.create("galactic;circle(0, 0, 1)")

    # using the regions package
    center = SkyCoord("0 deg", "0 deg", frame="galactic")
    region =  CircleSkyRegion(center=center, radius=1*u.deg)
    geom = RegionGeom(region)

    # Create a rectangular region with a 45 degree tilt
    # from DS9 string
    geom = RegionGeom.create("galactic;box(0, 0, 1,2,45)")

    # using the regions package
    region =  RectangleSkyRegion(center=center, width=1*u.deg, height=2*u.deg, angle=45*u.deg)
    geom = RegionGeom(region)

    # Equivalent factory method call
    geom = RegionGeom.create(region) 


Higher dimensional region geometries (cubes and hypercubes) can be constructed by passing a list of `~MapAxis` objects for non-spatial dimensions with the axes parameter:

.. code-block:: python

    from astropy.coordinates import SkyCoord
    from regions import CircleSkyRegion
    from gammapy.maps import MapAxis, RegionGeom
    import astropy.units as u

    energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')

    # from a DS9 string
    geom = RegionGeom.create("galactic;circle(0, 0, 1)", axes=[energy_axis])

    #using the regions package
    center = SkyCoord("0 deg", "0 deg", frame="galactic")
    region =  CircleSkyRegion(center=center, radius=1*u.deg)
    geom = RegionGeom(region, axes=[energy_axis])

The resulting `~RegionGeom` object has `ndim = 3`, two spatial dimensions with one single bin and the chosen energy axis with 12 bins:

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
A `~RegionGeom` defines a single spatial bin with arbitrary shape. The spatial coordinates are then given by the center of the region geometry. If one or more non-spatial axis are present, 
they can have any number of bins. There are different methods that can be used to access or modify the coordinates of a `~RegionGeom`.

+ Bin volume and angular size:
    The angular size of the region geometry is given by the method `~RegionGeom.solid_angle()`. If a region geometry has any number of non-spatial axes, 
    then the volume of each bin is given by `~RegionGeom.bin_volume()`.
    If there are no non-spatial axes, both return the same quantity.

    .. code-block:: python

        from gammapy.maps import MapAxis, RegionGeom

        # Create a circular region geometry with an energy axis
        energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
        geom = RegionGeom.create("galactic;circle(0, 0, 1)", axes=[energy_axis])

        # Get angular size and bin volume
        angular_size = geom.solid_angle()
        bin_volume = geom.bin_volume()
    
+ Coordinates defined by the `~RegionGeom`:
    Given a map coordinate or `~MapCoord` object, the method `~RegionGeom.contains()` checks if they are contained in the region geometry. One can also retrieve the coordinates of the region geometry with
    `~RegionGeom.get_coord()` and `~RegionGeom.get_idx()`, which return the sky coordinates and indexes respectively. Note that the spatial coordinate will always be a single entry, namely the center, while any non-spatial 
    axis can have as many bins as desired.

    .. code-block:: python

        from astropy.coordinates import SkyCoord
        from gammapy.maps import RegionGeom

        # Create a circular region geometry
        geom = RegionGeom.create("galactic;circle(0, 0, 1)")

        # Get coordinates and indexes defined by the region geometry
        coord = geom.get_coord()
        indexes = geom.get_idx()

        # Check if a coordinate is contained in the region
        >>> geom.contains(center)
            True
        # Check if an  array  of coordinates are contained in the region
        >>> coordinates = SkyCoord(l = [0, 0.5, 1.5], b = [0.5,2,0], frame='galactic', unit='deg')
        >>> geom.contains(coordinates)
            array([ True, False, False])
        


+ Upsampling and downsampling the non-spatial axes:
    The spatial binning of a `~RegionGeom` is made up of a single bin, that cannot be modified as it defines the region. However, if any non-spatial axes are present, they can be modified using the 
    `~RegionGeom.upsample()` and `~RegionGeom.downsample()` methods, which take as input a factor by which the indicated axis is to be up- or downsampled.

    .. code-block:: python

        from astropy.coordinates import SkyCoord
        from regions import CircleSkyRegion
        from gammapy.maps import MapAxis, RegionGeom
        import astropy.units as u

        # Create a circular region geometry with an energy axis
        energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
        geom = RegionGeom.create("galactic;circle(0, 0, 1)", axes=[energy_axis])

        # Upsample the energy axis by a factor 2
        geom_24_energy_bins = geom.upsample(2, "energy")
        # Downsample the energy axis by a factor 2
        geom_6_energy_bins = geom.downsample(2, "energy")

+ Image and WCS geometries:
    * If a `~RegionGeom` has any number of non-spatial axis, the corresponding region geometry with just the spatial dimensions is given by the method `~RegionGeom.to_image()`. If the region geometry only has spatial
      dimensions, a copy of it is returned.
    * Conversely, non-spatial axis can be added to an existing `~RegionGeom` by `~RegionGeom.to_cube()`, which takes a list of non-spatial axes with unique names to add to the region geometry.    
    * Region geometries are made of a single spatial bin, but are constructed on top of a finer `WcsGeom`. The method `~RegionGeom.to_wcs_geom()` returns the minimal equivalent geometry that contains the region geometry.
      It can also be given as an argument a minimal width for the resulting geometry.

    .. code-block:: python

        from gammapy.maps import MapAxis, RegionGeom

        # Create a circular region geometry
        geom = RegionGeom.create("galactic;circle(0, 0, 1)")

        # Add an energy axis to the region geometry
        energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
        geom_energy = geom.to_cube([energy_axis])

        # Get the image region geometry without energy axis. 
        # Note that geom_image == geom
        geom_image = geom_energy.to_image()

        >>> geom_image

            RegionGeom

            region     : CircleSkyRegion
            axes       : ['lon', 'lat']
            shape      : (1, 1)
            ndim       : 2
            frame      : galactic
            center     : 0.0 deg, 0.0 deg

        # Get the minimal wcs geometry that contains the region
        wcs_geom = geom.to_wcs_geom()

        >>> wcs_geom
            WcsGeom

            axes       : ['lon', 'lat']
            shape      : (202, 202)
            ndim       : 2
            frame      : galactic
            projection : TAN
            center     : 0.0 deg, 0.0 deg
            width      : 2.0 deg x 2.0 deg

Plotting a RegionGeom
---------------------
It can be useful to plot the region that defines a `~RegionGeom`, on its own or on top
of an existing `~Map`. This is done via `~RegionGeom.plot_region()`:

.. code-block:: python

    from gammapy.maps import RegionGeom
    geom = RegionGeom.create("icrs;circle(83.63, 22.01, 0.5)")
    geom.plot_region()

.. plot::

    from gammapy.maps import RegionGeom
    geom = RegionGeom.create("icrs;circle(83.63, 22.01, 0.5)")
    geom.plot_region()

One can also plot the region on top of an existing map, and change the properties of the
different regions by passing keyword arguments forwarded to `~regions.PixelRegion.as_artist`.

.. code-block:: python

    from gammapy.maps import RegionGeom, Map
    m = Map.create(width=3, skydir=(83.63, 22.01), frame='icrs')
    geom1 = RegionGeom.create("icrs;circle(83.63, 22.01, 0.5)")
    geom2 = RegionGeom.create("icrs;box(83.63, 22.01, 1,2,45)")
    m.plot(add_cbar=True)
    geom1.plot_region(ec="k")
    geom2.plot_region(lw=2, linestyle='--')

.. plot::

    from gammapy.maps import RegionGeom, Map
    m = Map.create(width=3, skydir=(83.63, 22.01), frame='icrs')
    geom1 = RegionGeom.create("icrs;circle(83.63, 22.01, 0.5)")
    geom2 = RegionGeom.create("icrs;box(83.63, 22.01, 1,2,45)")
    m.plot(add_cbar=True)
    geom1.plot_region(ec="k")
    geom2.plot_region(lw=2, linestyle='--')

RegionNDMap
===========
A `~RegionNDMap` owns a `~RegionGeom` instance as well as a data array containing map values.
It can be thought of as a `Map` but with a single spatial bin that can have an arbitrary 
shape, together with any non-spatial axis. It is to a `~RegionGeom` what a `~Map` is to a `~Geom`.

Creating a RegionNDMap
----------------------
A region map can be created either from a DS9 region string, an `regions.SkyRegion` object or an existing `~RegionGeom`:

.. code-block:: python

    from gammapy.maps import RegionGeom, RegionNDMap
    from astropy.coordinates import SkyCoord
    from regions import CircleSkyRegion
    import astropy.units as u

    # Create a map of a circular region with radius 0.5 deg centered around
    # the Crab Nebula

    # from DS9 string
    region_map = RegionNDMap.create("icrs;circle(83.63, 22.01, 0.5)")

    # using the regions package
    center = SkyCoord("83.63 deg", "22.01 deg", frame="icrs")
    region =  CircleSkyRegion(center=center, radius=0.5*u.deg)
    region_map = RegionNDMap.create(region)

    # from an existing RegionGeom, perhaps the one corresponding
    # to another, existing RegionNDMap
    geom = region_map.geom
    region_map_2 = RegionNDMap.from_geom(geom)

Higher dimensional region map objects (cubes and hypercubes) 
can be constructed by passing a list of `~MapAxis` objects for non-spatial dimensions with the axes parameter:

.. code-block:: python

    from gammapy.maps import MapAxis, RegionNDMap

    energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
    region_map = RegionNDMap.create("icrs;circle(83.63, 22.01, 0.5)", axes=[energy_axis])


Filling a RegionNDMap
---------------------

All the region maps created above are empty. In order to fill or access the data contained
in a `~RegionNDMap`, the `~RegionNDMap.data` attribute is used. In case the region map is being 
created from an existing `~RegionGeom`, this can be done in the same step:

.. code-block:: python

    from gammapy.maps import MapAxis, RegionNDMap
    import numpy as np  

    # Create a RegionNDMap with 12 energy bins
    energy_axis = MapAxis.from_bounds(100., 1e5, 12, interp='log', name='energy', unit='GeV')
    region_map = RegionNDMap.create("icrs;circle(83.63, 22.01, 0.5)", axes=[energy_axis])

    # Fill the region map
    #with the same value for each of the 12 energy bins
    region_map.data = 1 
    # or with an entry for each of the 12 energy bins
    region_map.data = np.linspace(1,50,12) 

    # Create another region map with the same RegionGeom but different data
    geom = region_map.geom
    region_map_2 = RegionNDMap.from_geom(geom, data = np.linspace(50,100,12))

    # Access the data
    print(region_map_2.data)

The data contained in a region map is a `~numpy.ndarray` with shape defined by the underlying
`~RegionGeom.data_shape`. In the case of only spatial dimensions, the shape is just (1,1), one single
spatial bin. If the associated `~RegionGeom` has a non-spatial axis with N bins, the data shape is
then (N, 1, 1), and similarly for additional non-spatial axes.

Visualization of a RegionNDMap
------------------------------
Visualizing a `~RegionNDMap` can be interpreted in two different ways. One is to plot a sky map that contains the region,
indicating the area of the sky encompassed by the spatial component of the region map. This is done via `~RegionNDMap.plot_region()`.
Another option is to plot the contents of the region map, which would be either a single value for the case of only spatial axes, 
or a function of the non-spatial axis bins. This is done by `~RegionNDMap.plot()` and `~RegionNDMap.plot_hist()`.

+ Plotting the underlying region:
    This is equivalent to the `~RegionGeom.plot_region()` described above, and, in fact, the `~RegionNDMap` method simply calls it on the associated 
    region geometry, `~RegionNDMap.geom`. Consequently, the use of this method is already described by the section above.

    .. code-block:: python

        from gammapy.maps import RegionNDMap
        region_map = RegionNDMap.create("icrs;circle(83.63, 22.01, 0.5)")
        region_map.plot_region()

+ Plotting the map contents:
    T
