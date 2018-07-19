"""
differential_evolution: The differential evolution global optimization algorithm

This file is largely (shamelessly) based on the Scipy's implementation of DE.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
import numbers

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps

def differential_evolution(func, bounds, args=(), strategy='best1bin',
                           maxiter=None, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube'):
    """Finds the global minimum of a multivariate function.

    Parameters
    ----------
    func : callable
        The wrapper for parallel invocation of the objective function to be
        minimized.  Must be in the form ``F(X, *args)``, where ``X`` is the
        argument in the form of a tuple of PyCUDA arrays and ``args`` is a
        tuple of any additional fixed parameters needed to completely specify
        the function. The objective function must be in the form ``f(x, *arg)``,
        where ``x`` is the argument taken from the i-th element from each
        arrays in ``X``, i.e. ``x = (X[0][i], ..., X[N][i])``.
    """
    pass

class DifferentialEvolutionSolver(object):
    pass
