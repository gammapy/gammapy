.. _regionmap:

The RegionGeom and RegionNDMap
==============================

This page provides examples and documentation specific to the Region
classes. 

RegionGeom
-----------
Like a geometry but instead of a small grid on a rectangular region, 
this is just one large pixel with an arbitrary shape that can have any 
number of non-spatial dimensions.



RegionNDMap
-----------
It's essentially a Map with one spatial bin that can have an arbitrary 
shape and whatever non-spatial axis you want.
It is to a RegionGeom what a Map is to a WcsGeom: it contains the data 
that goes into the grid.RegionGeom
----------