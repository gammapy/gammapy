


class ScaledRegularGridInterpolator(object):
    """
    Parameters
    ----------
    points : 

    values :

    values_scale : {'log', 'lin', 'sqrt'}
        Interpolation scaling applied to values. If the values vary over many magnitudes
        a 'log' scaling is recommended.
    **kwargs : dict
        Keyword arguments passed to `RegularGridInterpolator`.
    """

    def __init__(self, points, values, values_scale='linear', **kwargs):
        from scipy.interpolate import RegularGridInterpolator
        
        self.values_scale = values_scale

        kwargs.setdefault("bounds_error", False)
        kwargs.setdefault("method", "linear")
        
        if values_scale == "log":
            fn_0, fn_1 = np.log, np.exp
            kwargs.setdefault("fill_value", -np.inf)
        elif values_scale == "lin":
            fn_0, fn_1 = lambda x: x, lambda x: x
            kwargs.setdefault("fill_value", 0)
        elif values_scale == "sqrt":
            kwargs.setdefault("fill_value", 0)
            fn_0, fn_1 = np.sqrt, lambda x: x ** 2
        else:
            raise ValueError("Not a valid scaling mode.")
        
        self._fn0, self.fn_1 = fn_0, fn_1

        values_scaled = self._fn0(values)
        self._interpolate = RegularGridInterpolator(
                                    points=points,
                                    values=values_scaled,
                                    **kwargs)

    def __call__(self, **kwargs):
        values = self._interpolate(**kwargs)
        return self._fn1(values)