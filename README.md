# Python Implementation of Differential Evolution Algorithm with PyCUDA Arrays

_PyCUDE stands for Python-CUDA-DE._

## Description

PyCUDE is yet another Python package of parallel implementation of the
_Differential Evolution_ (DE) algorithm. The parallelization assumes that the
objective function is evaluated in parallel with parameters presented in PyCUDA
arrays.

The package is largely based on `scipy`'s implementation of
[DE](https://github.com/scipy/scipy/blob/master/scipy/optimize/_differentialevolution.py)
algorithm. The algorithm is invented by [Storn and Price](https://doi.org/10.1023/A:1008202821328).

## Installation
Clone and navigate into the repository, and then execute the command:

    python setup.py

This package depends on `scipy` and `numpy`.

## Usage
```python
differential_evolution(func, bounds, x0=None, args=(), strategy='best1bin',
maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7,
seed=None, callbacks=None, earlystop=None, disp=False, polish=False,
init='latinhypercube')
```

### Example

```python
from pycude import differential_evolution

# a function with input arguments in pycuda.gpuarray
def func(x, y):
    pass

bounds = [(0,2), (0, 2)]
result = differential_evolution(func, bounds)
```

### Parameters
* **func** : callable<br>
The wrapper for parallel invocation of the objective function to be
minimized.  Must be in the form ``F(X, *args)``, where ``X`` is the
argument in the form of a tuple of PyCUDA arrays and ``args`` is a
tuple of any additional fixed parameters needed to completely specify
the function. The objective function must be in the form ``f(x, *arg)``,
where ``x`` is the argument taken from the i-th element from each
arrays in ``X``, i.e. ``x = (X[0][i], ..., X[N][i])``.
* **bounds** : sequence<br>
Bounds for variables.  ``(min, max)`` pairs for each element in ``x``, defining
the lower and upper bounds for the optimizing argument of `func`. It is
required to have ``len(bounds) == len(x)``. ``len(bounds)`` is used to
determine the number of parameters in ``x``.
* **x0** : tuple, optional<br>
Initial value for ``x``. If ``x0`` is given, it will be placed at the
first element of the population.
* **args** : tuple, optional<br>
Any additional fixed parameters needed to completely specify the objective
function.
* **strategy** : str, optional<br>
The differential evolution strategy to use. Should be one of:
  - best1bin
  - best1exp
  - rand1exp
  - randtobest1exp
  - best2exp
  - rand2exp
  - randtobest1bin
  - best2bin
  - rand2bin
  - rand1bin

  The default is `best1bin`.
* **maxiter** : int, optional<br>
The maximum number of generations over which the entire population is
evolved. The maximum number of function evaluations (with no polishing)
is: ``(maxiter + 1) * popsize * len(x)``
* **popsize** : int, optional<br>
A multiplier for setting the total population size.  The population has
``popsize * len(x)`` individuals.
* **tol** : float, optional<br>
When the mean of the population energies, multiplied by tol,
divided by the standard deviation of the population energies
is greater than 1 the solving process terminates:
``convergence = mean(pop) * tol / stdev(pop) > 1``
* **mutation** : float or tuple(float, float), optional<br>
The mutation constant. In the literature this is also known as differential
weight, being denoted by ``F``. If specified as a float it should be in the
range [0, 2]. If specified as a tuple ``(min, max)`` dithering is employed.
Dithering randomly changes the mutation constant on a generation by generation
basis. The mutation constant for that generation is taken from ``U[min, max)``.
Dithering can help speed convergence significantly. Increasing the mutation
constant increases the search radius, but will slow down convergence.
* **recombination** : float, optional<br>
The recombination constant, should be in the range [0, 1]. In the literature
this is also known as the crossover probability, being denoted by CR. Increasing
this value allows a larger number of mutants to progress into the next
generation, but at the risk of population stability.
* **seed** : int or `np.random.RandomState`, optional<br>
If `seed` is not specified the `np.random.RandomState` singleton is used. If
`seed` is an int, a new `np.random.RandomState` instance is used, seeded with
`seed`. If `seed` is already a `np.random.RandomState` instance, then that
`np.random.RandomState` instance is used. Specify `seed` for repeatable
minimizations.
* **disp** : bool, optional<br>
Display status messages
* **earlystop** : callable, `earlystop(xk, convergence=val)`, optional<br>
A function to follow the progress of the minimization. ``xk`` is the current
value of ``x0``. ``val`` represents the fractional value of the population
convergence.  When ``val`` is greater than one the function halts. If
`earlystop` returns `True`, then the minimization is halted (any polishing is
still carried out).
* **callbacks** : a callable or a list of callables, optional<br>
A list of functions to be called at the end of each iteration. A callback will
be called with `callback(step=i, parameter=p, cost=c)`, where ``step`` is the
step of iteration, ``parameter`` is the best solution with the lowest ``cost``.
Note adding keyword arguments to the signature of a callback will avoid
'unexpected argument' error, i.e ``callback(..., **kwargs)``.
* **init** : string, optional<br>
Specify which type of population initialization is performed. Should be one of:
  - latinhypercube
  - random
