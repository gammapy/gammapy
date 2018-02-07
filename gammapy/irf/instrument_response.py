from astropy.table import Table
from gammapy.utils.nddata import BinnedDataAxis,  NDDataArray
import numpy as np



class InstrumentResponse(object):
    '''
    Class with an NDDataArray containing an instrument response.
    ...

    Attributes
    ----------
    names :  list of str
        Names of the axes of the NDDataArray
    data: NDDataArray
        The data array containing the insturment response

    Methods
    -------
    evaluate(point={'axis_name' : value})
        Evaulate the data and fixed point


    Examples
    -------
    Read and evaluate the effective area from a FITS file.
    >>> irf = InstrumentResponse.from_fits(path, extension='EFFECTIVE AREA')
    >>> point = {'THETA': 3.5*u.deg, 'ENERG': 1*u.TeV}
    >>> interpolated_result = irf.evaluate(point)

    '''

    def __init__(self, axes, data):
        self.data = NDDataArray(axes=axes, data=data)
        self.names = [ax.name for ax in axes]

    def evaluate(self, point):
        '''
        Evaluate the instrument response at the given point.

        Parameters
        -----------
        point: a dict containing the coordinates where you want to evaluate the response

        Returns
        -----------
        A number with an associated unit.


        Examples
        --------
        Evaluate instrument response at a point

        >>> point = {'DETX': 3.5*u.deg, 'DETY': 3.5*u.deg, 'ENERG': 1*u.TeV}
        >>> irf.evaluate(point)

        '''
        return self.data.evaluate(**point)

    @classmethod
    def from_fits(cls, path, extension='EFFECTIVE AREA', names=None, interpolation_modes=None):
        '''
        Read the instrument respone froma fits file given.

        Parameters
        -----------
        path: string
            path to a fits file containing the instrument response

        extension: string
            name of the HDU in the fits file which contains the instrument response

        names:list of strings
            Per default the column names of the bintable in the FITS file are used for axis names.

        interpolation_modes: list of strings
            TODO: No clue what that does for an NDDataArray

        Examples
        -----------
        >>> irf = InstrumentResponse.from_fits(path, extension='EFFECTIVE AREA')
        '''
        table = Table.read(path, hdu=extension)
        return cls.from_table(table, names)


    @classmethod
    def from_table(cls, table, names=None, interpolation_modes=None):
        '''
        Read the instrument respone from an astropy table

        Parameters
        -----------
        table: astropy.table.Table
            the table contianing the instrument response

        names:list of strings
            Per default the column names of the table are used for axis names.

        interpolation_modes: list of strings
            TODO: No clue what that does for an NDDataArray

        Examples
        -----------
        >>> table = Table.read(path)
        >>> irf = InstrumentResponse.from_table(table)
        '''
        number_of_columns = len(table.colnames)

        if not interpolation_modes:
            interpolation_modes = ['linear', ] * (number_of_columns - 1)

        # read table data that srores n-dimensional data in ogip convention
        bounds = table.colnames[:-1]
        low_bounds = bounds[::2]
        high_bounds = bounds[1::2]

        data = table[table.colnames[-1]].quantity[0].T

        # we need to check this unit specifically because ctools writes weird units into fits tables and astropy goes haywire
        if data.unit == '1/s/MeV/sr':
            import astropy.units as u
            data = data.value * u.Unit('1/(s MeV sr)')

        if not names:
            names = [n.replace('_LO', '') for n in low_bounds]

        axes = []
        for colname_low, colname_high, name, mode in zip(low_bounds, high_bounds, names, interpolation_modes):
            low = np.ravel(table[colname_low]).quantity
            high = np.ravel(table[colname_high]).quantity

            axes.append(BinnedDataAxis(low, high, interpolation_mode=mode, name=name))

        return cls(axes=axes, data=data)
