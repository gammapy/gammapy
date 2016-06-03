# Licensed under a 3-clause BSD style license - see LICENSE.rst

import string
import itertools
import re
from astropy import units as u
from astropy import coordinates
from astropy.coordinates import BaseCoordinateFrame
from astropy import log
from ..shapes import circle, rectangle, polygon, ellipse, point
from ..core import PixCoord

__all__ = ['read_ds9']

def read_ds9(filename):
    """
    Read a ds9 region file in as a list of astropy region objects

    Parameters
    ----------
    filename : str
        The file path

    Returns
    -------
    A list of region objects
    """
    region_list = ds9_parser(filename)
    return region_list_to_objects(region_list)


def parse_coordinate(string_rep, unit):
    """
    Parse a single coordinate
    """
    # Any ds9 coordinate representation (sexagesimal or degrees)
    if 'd' in string_rep or 'h' in string_rep:
        return coordinates.Angle(string_rep)
    elif unit is 'hour_or_deg':
        if ':' in string_rep:
            spl = tuple([float(x) for x in string_rep.split(":")])
            return coordinates.Angle(spl, u.hourangle)
        else:
            ang = float(string_rep)
            return coordinates.Angle(ang, u.deg)
    elif unit.is_equivalent(u.deg):
        #return coordinates.Angle(string_rep, unit=unit)
        if ':' in string_rep:
            ang = tuple([float(x) for x in string_rep.split(":")])
        else:
            ang = float(string_rep)
        return coordinates.Angle(ang, u.deg)
    else:
        return u.Quantity(float(string_rep), unit)

unit_mapping = {'"': u.arcsec,
                "'": u.arcmin,
                'r': u.rad,
                'i': u.dimensionless_unscaled,
               }


def parse_angular_length_quantity(string_rep):
    """
    Given a string that is either a number or a number and a unit, return a
    Quantity of that string.  e.g.:

        23.9 -> 23.9*u.deg
        50" -> 50*u.arcsec
    """
    has_unit = string_rep[-1] not in string.digits
    if has_unit:
        unit = unit_mapping[string_rep[-1]]
        return u.Quantity(float(string_rep[:-1]), unit=unit)
    else:
        return u.Quantity(float(string_rep), unit=u.deg)

# these are the same function, just different names
radius = parse_angular_length_quantity
width = parse_angular_length_quantity
height = parse_angular_length_quantity
angle = parse_angular_length_quantity

# For the sake of readability in describing the spec, parse_coordinate etc. are renamed here
coordinate = parse_coordinate
language_spec = {'point': (coordinate, coordinate),
                 'circle': (coordinate, coordinate, radius),
                 # This is a special case to deal with n elliptical annuli
                 'ellipse': itertools.chain((coordinate, coordinate), itertools.cycle((radius, ))),
                 'box': (coordinate, coordinate, width, height, angle),
                 'polygon': itertools.cycle((coordinate, )),
                }

coordinate_systems = ['fk5', 'fk4', 'icrs', 'galactic', 'wcs', 'physical', 'image', 'ecliptic']
coordinate_systems += ['wcs{0}'.format(letter) for letter in string.ascii_lowercase]

coordsys_name_mapping = dict(zip(coordinates.frame_transform_graph.get_names(),
                                 coordinates.frame_transform_graph.get_names()))
coordsys_name_mapping['ecliptic'] = 'geocentrictrueecliptic' # needs expert attention TODO

hour_or_deg = 'hour_or_deg'
coordinate_units = {'fk5': (hour_or_deg, u.deg),
                    'fk4': (hour_or_deg, u.deg),
                    'icrs': (hour_or_deg, u.deg),
                    'geocentrictrueecliptic': (u.deg, u.deg),
                    'galactic': (u.deg, u.deg),
                    'physical': (u.dimensionless_unscaled, u.dimensionless_unscaled),
                    'image': (u.dimensionless_unscaled, u.dimensionless_unscaled),
                    'wcs': (u.dimensionless_unscaled, u.dimensionless_unscaled),
                   }
for letter in string.ascii_lowercase:
    coordinate_units['wcs{0}'.format(letter)] = (u.dimensionless_unscaled, u.dimensionless_unscaled)

region_type_or_coordsys_re = re.compile("^#? *(-?)([a-zA-Z0-9]+)")

paren = re.compile("[()]")

def strip_paren(string_rep):
    return paren.sub("", string_rep)


def region_list_to_objects(region_list):
    """
    Given a list of parsed region tuples, product a list of astropy objects
    """
    viz_keywords = ['color', 'dashed', 'width', 'point', 'font', 'text']

    output_list = []
    for region_type, coord_list, meta in region_list:

        # TODO: refactor, possible on the basis of # of parameters + sometimes handle corner cases

        if region_type == 'circle':
            if isinstance(coord_list[0], BaseCoordinateFrame):
                reg = circle.CircleSkyRegion(coord_list[0], coord_list[1])
            elif isinstance(coord_list[0], PixCoord):
                reg = circle.CirclePixelRegion(coord_list[0], coord_list[1])
            else:
                raise ValueError("No central coordinate")
        elif region_type == 'ellipse':
            # Do not read elliptical annuli for now
            if len(coord_list) > 4:
                continue
            if isinstance(coord_list[0], BaseCoordinateFrame):
                reg = ellipse.EllipseSkyRegion(coord_list[0], coord_list[1], coord_list[2], coord_list[3])
            elif isinstance(coord_list[0], PixCoord):
                reg = ellipse.EllipsePixelRegion(coord_list[0], coord_list[1], coord_list[2], coord_list[3])
            else:
                raise ValueError("No central coordinate")
        elif region_type == 'polygon':
            if isinstance(coord_list[0], BaseCoordinateFrame):
                reg = polygon.PolygonSkyRegion(coord_list[0])
            elif isinstance(coord_list[0], PixCoord):
                reg = polygon.PolygonPixelRegion(coord_list[0])
            else:
                raise ValueError("No central coordinate")
        elif region_type == 'rectangle':
            if isinstance(coord_list[0], BaseCoordinateFrame):
                reg = rectangle.RectangleSkyRegion(coord_list[0], coord_list[1], coord_list[2], coord_list[3])
            elif isinstance(coord_list[0], PixCoord):
                reg = rectangle.RectanglePixelRegion(coord_list[0], coord_list[1], coord_list[2], coord_list[3])
            else:
                raise ValueError("No central coordinate")
        elif region_type == 'point':
            if isinstance(coord_list[0], BaseCoordinateFrame):
                reg = point.PointSkyRegion(coord_list[0])
            elif isinstance(coord_list[0], PixCoord):
                reg = point.PointPixelRegion(coord_list[0])
            else:
                raise ValueError("No central coordinate")
        else:
            continue
        reg.vizmeta = {key: meta[key] for key in meta.keys() if key in viz_keywords}
        reg.meta = {key: meta[key] for key in meta.keys() if key not in viz_keywords}
        output_list.append(reg)
    return output_list


def ds9_parser(filename):
    """
    Parse a complete ds9 .reg file

    Returns
    -------
    list of (region type, coord_list, meta, composite, include) tuples
    region_type : str
    coord_list : list of coordinate objects
    meta : metadata dict
    composite : bool
        indicates whether region is a composite region
    include : bool
        Whether the region is included (False -> excluded)
    """
    coordsys = None
    regions = []
    composite_region = None

    with open(filename,'r') as fh:
        for line_ in fh:
            # ds9 regions can be split on \n or ;
            for line in line_.split(";"):
                parsed = line_parser(line, coordsys)
                if parsed in coordinate_systems:
                    coordsys = parsed
                elif parsed:
                    region_type, coordlist, meta, composite, include = parsed
                    meta['include'] = include
                    log.debug("Region type = {0}".format(region_type))
                    if composite and composite_region is None:
                        composite_region = [(region_type, coordlist)]
                    elif composite:
                        composite_region.append((region_type, coordlist))
                    elif composite_region is not None:
                        composite_region.append((region_type, coordlist))
                        regions.append(composite_region)
                        composite_region = None
                    else:
                        regions.append((region_type, coordlist, meta))

    return regions


def line_parser(line, coordsys=None):
    """
    Parse a single ds9 region line into a string

    Parameters
    ----------
    line : str
        A single ds9 region contained in a string
    coordsys : str
        The global coordinate system name declared at the top of the ds9 file

    Returns
    -------
    (region_type, parsed_return, parsed_meta, composite, include)
    region_type : str
    coord_list : list of coordinate objects
    meta : metadata dict
    composite : bool
        indicates whether region is a composite region
    include : bool
        Whether the region is included (False -> excluded)
    """
    region_type_search = region_type_or_coordsys_re.search(line)
    if region_type_search:
        include = region_type_search.groups()[0]
        region_type = region_type_search.groups()[1]
    else:
        return

    if region_type in coordinate_systems:
        return region_type # outer loop has to do something with the coordinate system information
    elif region_type in language_spec:
        if coordsys is None:
            raise ValueError("No coordinate system specified and a region has been found.")

        if "||" in line:
            composite = True
        else:
            composite = False

        # end_of_region_name is the coordinate of the end of the region's name, e.g.:
        # circle would be 6 because circle is 6 characters
        end_of_region_name = region_type_search.span()[1]
        # coordinate of the # symbol or end of the line (-1) if not found
        hash_or_end = line.find("#")
        coords_etc = strip_paren(line[end_of_region_name:hash_or_end].strip(" |"))
        meta_str = line[hash_or_end:]

        parsed_meta = meta_parser(meta_str)

        if coordsys in coordsys_name_mapping:
            parsed = type_parser(coords_etc, language_spec[region_type],
                                 coordsys_name_mapping[coordsys])

            # Reset iterator for ellipse annulus
            if region_type == 'ellipse':
                language_spec[region_type] = itertools.chain((coordinate, coordinate), itertools.cycle((radius, )))

            parsed_angles = [(x, y) for x, y in zip(parsed[:-1:2],
                                                    parsed[1::2])
                             if isinstance(x, coordinates.Angle) and
                             isinstance(x, coordinates.Angle)]
            frame = coordinates.frame_transform_graph.lookup_name(coordsys_name_mapping[coordsys])

            lon,lat = zip(*parsed_angles)
            lon, lat = u.Quantity(lon), u.Quantity(lat)
            sphcoords = coordinates.UnitSphericalRepresentation(lon, lat)
            coords = frame(sphcoords)

            return region_type, [coords] + parsed[len(coords)*2:], parsed_meta, composite, include
        else:
            parsed = type_parser(coords_etc, language_spec[region_type],
                                 coordsys)
            if region_type == 'polygon':
                # have to special-case polygon in the phys coord case b/c can't typecheck when iterating as in sky coord case
                coord = PixCoord(parsed[0::2], parsed[1::2])
                parsed_return = [coord]
            else:
                parsed = [_.value for _ in parsed]
                coord = PixCoord(parsed[0], parsed[1])
                parsed_return = [coord]+parsed[2:]

            # Reset iterator for ellipse annulus
            if region_type == 'ellipse':
                language_spec[region_type] = itertools.chain((coordinate, coordinate), itertools.cycle((radius, )))

            return region_type, parsed_return, parsed_meta, composite, include


def type_parser(string_rep, specification, coordsys):
    """
    For a given region line in which the type has already been determined,
    parse the coordinate definition

    Parameters
    ----------
    string_rep : str
        The string containing the coordinates.  For example, if your region is
        `circle(1,2,3)` this string would be `(1,2,3)`
    specification : iterable
        An iterable of coordinate specifications.  For example, for a circle,
        this would be a list of (coordinate, coordinate, radius).  Each
        individual specification should be a function that takes a string and
        returns the appropriate astropy object.  See ``language_spec`` for the
        definition of the grammar used here.
    coordsys : str
        The string name of the global coordinate system

    Returns
    -------
    coord_list : list
        The list of astropy coordinates and/or quantities representing radius,
        width, etc. for the region
    """
    coord_list = []
    splitter = re.compile("[, ]")
    for ii, (element, element_parser) in enumerate(zip(splitter.split(string_rep), specification)):
        if element_parser is coordinate:
            unit = coordinate_units[coordsys][ii % 2]
            coord_list.append(element_parser(element, unit))
        else:
            coord_list.append(element_parser(element))

    return coord_list


# match an x=y pair (where y can be any set of characters) that may or may not
# be followed by another one
meta_token = re.compile("([a-zA-Z]+)(=)([^= ]+) ?")

#meta_spec = {'color': color,
#            }
# global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
# ruler(+175:07:14.900,+50:56:21.236,+175:06:52.643,+50:56:11.190) ruler=physical physical color=white font="helvetica 12 normal roman" text={Ruler}
    


def meta_parser(meta_str):
    """
    Parse the metadata for a single ds9 region string.  The metadata is
    everything after the close-paren of the region coordinate specification.
    All metadata is specified as key=value pairs separated by whitespace, but
    sometimes the values can also be whitespace separated.
    """
    meta_token_split = [x for x in meta_token.split(meta_str.strip()) if x]
    equals_inds = [i for i, x in enumerate(meta_token_split) if x is '=']
    result = {meta_token_split[ii-1]:
              " ".join(meta_token_split[ii+1:jj-1 if jj is not None else None])
              for ii,jj in zip(equals_inds, equals_inds[1:]+[None])}

    return result
