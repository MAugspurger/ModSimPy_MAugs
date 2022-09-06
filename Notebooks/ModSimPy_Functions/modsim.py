"""
Code from Modeling and Simulation in Python.

Copyright 2020 Allen Downey

MIT License: https://opensource.org/licenses/MIT
"""

import logging

logger = logging.getLogger(name="modsim.py")

# make sure we have Python 3.6 or better
import sys

if sys.version_info < (3, 6):
    logger.warning("modsim.py depends on Python 3.6 features.")

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import scipy.optimize as spo

from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.integrate import solve_ivp

from types import SimpleNamespace
from copy import copy


def flip(p=0.5):
    """Flips a coin with the given probability.

    p: float 0-1

    returns: boolean (True or False)
    """
    return np.random.random() < p


from numpy import linspace


def root_scalar(func, *args, **kwargs):
    """Finds the input value that minimizes `min_func`.

    Wrapper for
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html

    func: computes the function to be minimized
    bracket: sequence of two values, lower and upper bounds of the range to be searched
    args: any additional positional arguments are passed to func
    kwargs: any keyword arguments are passed to root_scalar

    returns: RootResults object
    """
    bracket = kwargs.get('bracket', None)
    if bracket is None or len(bracket) != 2:
        msg = ("To run root_scalar, you have to provide a "
               "`bracket` keyword argument with a sequence "
               "of length 2.")
        raise ValueError(msg)

    try:
        func(bracket[0], *args)
    except Exception as e:
        msg = ("Before running scipy.integrate.root_scalar "
               "I tried running the function you provided "
               "with `bracket[0]`, "
               "and I got the following error:")
        logger.error(msg)
        raise (e)

    underride(kwargs, rtol=1e-4)

    res = spo.root_scalar(func, *args, **kwargs)

    if not res.converged:
        msg = ("scipy.optimize.root_scalar did not converge. "
               "The message it returned is:\n" + res.flag)
        raise ValueError(msg)

    return res


def minimize_scalar(func, *args, **kwargs):
    """Finds the input value that minimizes `func`.

    Wrapper for
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    func: computes the function to be minimized
    args: any additional positional arguments are passed to func
    kwargs: any keyword arguments are passed to minimize_scalar

    returns: OptimizeResult object
    """
    bounds = kwargs.get('bounds', None)

    if bounds is None or len(bounds) != 2:
        msg = ("To run maximize_scalar or minimize_scalar, "
               "you have to provide a `bounds` "
               "keyword argument with a sequence "
               "of length 2.")
        raise ValueError(msg)

    try:
        func(bounds[0], *args)
    except Exception as e:
        msg = ("Before running scipy.integrate.minimize_scalar, "
               "I tried running the function you provided "
               "with the lower bound, "
               "and I got the following error:")
        logger.error(msg)
        raise (e)

    underride(kwargs, method='bounded')

    res = spo.minimize_scalar(func, args=args, **kwargs)

    if not res.success:
        msg = ("minimize_scalar did not succeed."
               "The message it returned is: \n" +
               res.message)
        raise Exception(msg)

    return res


def maximize_scalar(max_func, *args, **kwargs):
    """Finds the input value that maximizes `max_func`.

    Wrapper for https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    min_func: computes the function to be maximized
    args: any additional positional arguments are passed to max_func
    options: any keyword arguments are passed as options to minimize_scalar

    returns: ModSimSeries object
    """
    def min_func(*args):
        return -max_func(*args)

    res = minimize_scalar(min_func, *args, **kwargs)

    # we have to negate the function value before returning res
    res.fun = -res.fun
    return res


def run_solve_ivp(system, slope_func, **options):
    """Computes a numerical solution to a differential equation.

    `system` must contain `init` with initial conditions,
    `t_end` with the end time.  Optionally, it can contain
    `t_0` with the start time.

    It should contain any other parameters required by the
    slope function.

    `options` can be any legal options of `scipy.integrate.solve_ivp`

    system: dictionary object
    slope_func: function that computes slopes

    returns: TimeFrame
    """

    # make sure `system` contains `init`
    if 'init' not in system:
        msg = """It looks like the dictionary`system` does not contain `init`
                 as a key.  `init` should be a state pd.Series
                 object that specifies the initial condition:"""
        raise ValueError(msg)

    # make sure `system` contains `t_end`
    if 't_end' not in system:
        msg = """It looks like `system` does not contain `t_end`
                 as a key.  `t_end` should be the
                 final time:"""
        raise ValueError(msg)

    # the default value for t_0 is 0
    if 't_0' not in system:
        system['t_0'] = 0

    # try running the slope function with the initial conditions
    try:
        slope_func(system['t_0'], system['init'], system)
    except Exception as e:
        msg = """Before running scipy.integrate.solve_ivp, I tried
                 running the slope function you provided with the
                 initial conditions in `system` and `t=t_0` and I got
                 the following error:"""
        logger.error(msg)
        raise (e)

    # get the list of event functions
    events = options.get('events', [])

    # if there's only one event function, put it in a list
    try:
        iter(events)
    except TypeError:
        events = [events]

    for event_func in events:
        # make events terminal unless otherwise specified
        if not hasattr(event_func, 'terminal'):
            event_func.terminal = True

        # test the event function with the initial conditions
        try:
            event_func(system['t_0'], system['init'], system)
        except Exception as e:
            msg = """Before running scipy.integrate.solve_ivp, I tried
                     running the event function you provided with the
                     initial conditions in `system` and `t=t_0` and I got
                     the following error:"""
            logger.error(msg)
            raise (e)

    # get dense output (i.e. a continuous solution) unless otherwise specified
    if not 'dense_output' in options:
        options['dense_output'] = True

    # run the solver
    # a 'bunch' object is a dictionary whose values can be accessed
    # using attribute style syntax  (i.e. bunch1.t_0 rather than dict1['t_0'])
    bunch = solve_ivp(slope_func, [system['t_0'], system['t_end']], system['init'],
                      args=[system], **options)
    
    # 'bunch' contains numpy ndarrays called y and t.  
    # y contains the results for state of the system over
    #  the whole simulation.  t contains the time
    # steps.  The following lines separate them into separate
    # arrays
    y = bunch.y
    t = bunch.t

    # get the column names from `init`
    columns = system['init'].keys()

    # evaluate the results at equally-spaced points
    # the first code block runs in t_eval is defined or if dense_output is false
    if 't_eval' in options or not options['dense_output']:
        results = pd.DataFrame(y.T, index=t,
                        columns=columns)
    else:
        # Assign the number of time steps for dense_output = True option
        num = 101
        # Define the solution at each of the time steps
        t_final = t[-1]
        t_array = linspace(system['t_0'], t_final, num)
        y_array = bunch.sol(t_array)
        # pack the results into a DataFrame
        results = pd.DataFrame(y_array.T, index=t_array,
                                   columns=columns)

    return results, bunch


def leastsq(error_func, x0, *args, **options):
    """Find the parameters that yield the best fit for the data.

    `x0` can be a sequence, array, Series, or Params

    Positional arguments are passed along to `error_func`.

    Keyword arguments are passed to `scipy.optimize.leastsq`

    error_func: function that computes a sequence of errors
    x0: initial guess for the best parameters
    args: passed to error_func
    options: passed to leastsq

    :returns: Params object with best_params and ModSimSeries with details
    """
    # override `full_output` so we get a message if something goes wrong
    options["full_output"] = True
        

    # run leastsq
    t = scipy.optimize.leastsq(error_func, x0=x0, args=args, **options)
    best_params, cov_x, infodict, mesg, ier = t

    # pack the results into a ModSimSeries object
    details = SimpleNamespace(cov_x=cov_x,
                              mesg=mesg,
                              ier=ier,
                              **infodict)
    details.success = details.ier in [1,2,3,4]

    # return the best parameters and details
    return best_params, details


def has_nan(a):
    """Checks whether the an array contains any NaNs.

    :param a: NumPy array or Pandas Series
    :return: boolean
    """
    return np.any(np.isnan(a))


def is_strictly_increasing(a):
    """Checks whether the elements of an array are strictly increasing.

    :param a: NumPy array or Pandas Series
    :return: boolean
    """
    return np.all(np.diff(a) > 0)


def source_code(obj):
    """Prints the source code for a given object.

    obj: function or method object
    """
    print(inspect.getsource(obj))


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d


def contour(df,**options):
    """Makes a contour plot from a DataFrame.

    Wrapper for plt.contour
    https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.contour.html

    Note: columns and index must be numerical

    df: DataFrame
    options: passed to plt.contour
    """
    fontsize = options.pop("fontsize", 12)
    underride(options, cmap="viridis")
    x = df.columns
    y = df.index
    X, Y = np.meshgrid(x, y)
    cs = plt.contour(X, Y, df, **options)
    plt.clabel(cs, inline=1, fontsize=fontsize)


def savefig(filename, **options):
    """Save the current figure.

    Keyword arguments are passed along to plt.savefig

    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.savefig.html

    filename: string
    """
    print("Saving figure to file", filename)
    plt.savefig(filename, **options)



def magnitude(x):
    """Returns the magnitude of a Quantity or number.

    x: Quantity or number

    returns: number
    """
    return x.magnitude if hasattr(x, 'magnitude') else x


def Vector(x, y, z=None, **options):
    """
    """
    underride(options, name='component')
    if z is None:
        return pd.Series(dict(x=x, y=y), **options)
    else:
        return pd.Series(dict(x=x, y=y, z=z), **options)


## Vector functions (should work with any sequence)

def vector_mag(v):
    """Vector magnitude."""
    return np.sqrt(np.dot(v, v))


def vector_mag2(v):
    """Vector magnitude squared."""
    return np.dot(v, v)


def vector_angle(v):
    """Angle between v and the positive x axis.

    Only works with 2-D vectors.

    returns: angle in radians
    """
    assert len(v) == 2
    x, y = v
    return np.arctan2(y, x)


def vector_polar(v):
    """Vector magnitude and angle.

    returns: (number, angle in radians)
    """
    return vector_mag(v), vector_angle(v)


def vector_hat(v):
    """Unit vector in the direction of v.

    returns: Vector or array
    """
    # check if the magnitude of the Quantity is 0
    mag = vector_mag(v)
    if mag == 0:
        return v
    else:
        return v / mag


def vector_perp(v):
    """Perpendicular Vector (rotated left).

    Only works with 2-D Vectors.

    returns: Vector
    """
    assert len(v) == 2
    x, y = v
    return Vector(-y, x)


def vector_dot(v, w):
    """Dot product of v and w.

    returns: number or Quantity
    """
    return np.dot(v, w)


def vector_cross(v, w):
    """Cross product of v and w.

    returns: number or Quantity for 2-D, Vector for 3-D
    """
    res = np.cross(v, w)

    if len(v) == 3:
        return Vector(*res)
    else:
        return res


def vector_proj(v, w):
    """Projection of v onto w.

    returns: array or Vector with direction of w and units of v.
    """
    w_hat = vector_hat(w)
    return vector_dot(v, w_hat) * w_hat


def scalar_proj(v, w):
    """Returns the scalar projection of v onto w.

    Which is the magnitude of the projection of v onto w.

    returns: scalar with units of v.
    """
    return vector_dot(v, vector_hat(w))


def vector_dist(v, w):
    """Euclidean distance from v to w, with units."""
    if isinstance(v, list):
        v = np.asarray(v)
    return vector_mag(v - w)


def vector_diff_angle(v, w):
    """Angular difference between two vectors, in radians.
    """
    if len(v) == 2:
        return vector_angle(v) - vector_angle(w)
    else:
        # TODO: see http://www.euclideanspace.com/maths/algebra/
        # vectors/angleBetween/
        raise NotImplementedError()

def pol2cart(theta, rho, z=None):
    """Convert polar coordinates to Cartesian.

    theta: number or sequence in radians
    rho: number or sequence
    z: number or sequence (optional)

    returns: x, y OR x, y, z
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    if z is None:
        return x, y
    else:
        return x, y, z
        
def plot_segment(A, B, **options):
    """Plots a line segment between two Vectors.

    For 3-D vectors, the z axis is ignored.

    Additional options are passed along to plot().

    A: Vector
    B: Vector
    """
    xs = A.x, B.x
    ys = A.y, B.y
    plt.plot(xs, ys, **options)
