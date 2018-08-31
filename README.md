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

## Example
```python
from pycude import differential_evolution

# a function with input arguments in pycuda.gpuarray
def func(x, y):
    pass

bounds = [(0,2), (0, 2)]
result = differential_evolution(func, bounds)
```

## Documentation
Please access the documentation [here](https://chungheng.github.io/pycude/).
