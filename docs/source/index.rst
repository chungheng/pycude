.. -*- rst -*-

PyCUDE
======

PyCUDE is yet another package of parallel implementation for the
`Differential Evolution <https://en.wikipedia.org/wiki/Differential_evolution>`_
algorithm. The parallelization assumes that the objective function
can be evaluated in parallel with parameters presented in PyCUDA arrays.

This package is largely based on `scipy`'s implementation of
`DE <https://github.com/scipy/scipy/blob/master/scipy/optimize/_differentialevolution.py>`_
algorithm. The algorithm is invented by `Storn and Price <https://doi.org/10.1023/A:1008202821328>`_.

Installation
------------
Clone and navigate into the repository, and then execute the command:
::

    python setup.py

This package depends on `scipy` and `numpy`.

Example
-------
.. code-block:: python

   from pycude import differential_evolution

   # a function with input arguments in pycuda.gpuarray
   def func(x, y):
       pass

   bounds = [(0,2), (0, 2)]
   result = differential_evolution(func, bounds)


.. toctree::
   :maxdepth: 2

   reference
   license


Index
-----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
