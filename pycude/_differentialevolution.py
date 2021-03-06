"""
differential_evolution: The differential evolution global optimization algorithm

This file is largely (shamelessly) based on the Scipy's implementation of DE.
"""
from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
import numbers

try:
    import pycuda.gpuarray as garray
    import pycuda.driver as gdrv
except:
    pass

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps

def differential_evolution(func, bounds, x0=None, args=(), strategy='best1bin',
                           maxiter=None, popsize=0, popscale=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callbacks=None, earlystop=None, disp=False, polish=False, init='latinhypercube'):
    """Finds the global minimum of a multivariate function.
    This implementation is largely based on Scipy's implementation of DE.

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
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    x0 : tuple, optional
        Initial value for ``x``. If ``x0`` is given, it will be placed at the
        first element of the population.
    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.
    strategy : str, optional
        The differential evolution strategy to use. Should be one of:
            - 'best1bin'
            - 'best1exp'
            - 'rand1exp'
            - 'randtobest1exp'
            - 'best2exp'
            - 'rand2exp'
            - 'randtobest1bin'
            - 'best2bin'
            - 'rand2bin'
            - 'rand1bin'
        The default is 'best1bin'
    maxiter : int, optional
        The maximum number of generations over which the entire population is
        evolved. The maximum number of function evaluations (with no polishing)
        is: ``(maxiter + 1) * popsize``.
    popsize : int, optional
        Population size. If zero, the population size is set with ``popscale``.
    popscale : int, optional
        A multiplier for setting the total population size.  The population has
        ``popsize = popscale * len(x)`` individuals.
    tol : float, optional
        When the mean of the population energies, multiplied by tol,
        divided by the standard deviation of the population energies
        is greater than 1 the solving process terminates:
        ``convergence = mean(pop) * tol / stdev(pop) > 1``.
    mutation : float or tuple(float, float), optional
        The mutation constant. In the literature this is also known as
        differential weight, being denoted by F.
        If specified as a float it should be in the range [0, 2].
        If specified as a tuple ``(min, max)`` dithering is employed. Dithering
        randomly changes the mutation constant on a generation by generation
        basis. The mutation constant for that generation is taken from
        U[min, max). Dithering can help speed convergence significantly.
        Increasing the mutation constant increases the search radius, but will
        slow down convergence.
    recombination : float, optional
        The recombination constant, should be in the range [0, 1]. In the
        literature this is also known as the crossover probability, being
        denoted by CR. Increasing this value allows a larger number of mutants
        to progress into the next generation, but at the risk of population
        stability.
    seed : int or `np.random.RandomState`, optional
        If `seed` is not specified the `np.random.RandomState` singleton is
        used.
        If `seed` is an int, a new `np.random.RandomState` instance is used,
        seeded with `seed`.
        If `seed` is already a `np.random.RandomState` instance, then that
        `np.random.RandomState` instance is used.
        Specify `seed` for repeatable minimizations.
    disp : bool, optional
        Display status messages
    earlystop : callable, `earlystop(xk, convergence=val)`, optional
        A function to follow the progress of the minimization. ``xk`` is
        the current value of ``x0``. ``val`` represents the fractional
        value of the population convergence.  When ``val`` is greater than one
        the function halts. If `earlystop` returns `True`, then the minimization
        is halted (any polishing is still carried out).
    callbacks : a callable or a list of callables, optional
        A list of functions to be called at the end of each iteration. A
        callback will be called with `callback(step=i, parameter=p, cost=c)`,
        where ``step`` is the step of iteration, ``parameter`` is the best
        solution with the lowest ``cost``. Note adding keyword arguments to the
        signature of a callback will avoid 'unexpected argument' error, i.e
        ``callback(..., **kwargs)``.
    polish : bool, optional
        If True, then `scipy.optimize.minimize` with the `L-BFGS-B` method
        is used to polish the best population member at the end. This requires
        a few more function evaluations.
    init : string, optional
        Specify which type of population initialization is performed. Should be
        one of:
            - 'latinhypercube'
            - 'random'
    """
    solver = DifferentialEvolutionSolver(func, bounds, args=args, x0=None,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, popscale=popscale, tol=tol, mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         earlystop=earlystop,
                                         callbacks=callbacks,
                                         disp=disp,
                                         init=init)
    return solver.solve()

class DifferentialEvolutionSolver(object):

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    def __init__(self, func, bounds, args=(), x0=None,
                 strategy='best1bin', maxiter=None, popsize=0, popscale=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 callbacks=None, earlystop=None, disp=False, polish=False,
                 init='latinhypercube'):

        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        if callbacks is not None and callable(callbacks):
            self.callbacks = (callbacks,)
        else:
            self.callbacks = callbacks
        self.earlystop = earlystop
        self.polish = polish
        self.tol = tol

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2
                or not np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        self.maxiter = maxiter or 1000

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = _make_random_gen(seed)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        self.num_population_members = popsize or popscale * self.parameter_count

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        if init == 'latinhypercube':
            self.init_population_lhs()
        elif init == 'random':
            self.init_population_random()
        else:
            raise ValueError("The population initialization method must be one"
                             "of 'latinhypercube' or 'random'")

        self.population_energies = (np.ones(self.num_population_members)
                                    * np.inf)

        self.disp = disp

        self.init_pycuda_arrays()

        if x0 is not None:
            p0 = self._unscale_parameters(x0)
            self.population[0][:] = x0


    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.random_sample(self.population_shape)

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)

    def init_pycuda_arrays(self):
        dtype = self.population.dtype
        self.gpu_arrays = []

        for i in range(self.parameter_count):
            array = garray.zeros(self.num_population_members, dtype=dtype)
            self.gpu_arrays.append(array)

    @property
    def x(self):
        """
        The best solution from the solver
        Returns
        -------
        x - ndarray
            The best solution from the solver.
        """
        return self._scale_parameters(self.population[0])

    def evaluate_func(self, parameters):
        for index, (dest, src) in enumerate(zip(self.gpu_arrays, parameters.T)):
            gdrv.memcpy_htod(dest.gpudata, src)

        return self.func(self.gpu_arrays, *self.args)

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """

        nfev, nit, warning_flag = 0, 0, False
        status_message = _status_message['success']

        # calculate energies to start with
        parameters = np.zeros_like(self.population, order='F')
        for index, candidate in enumerate(self.population):
            parameters[index, :] = self._scale_parameters(candidate)

        self.population_energies[:] = self.evaluate_func(parameters)
        nfev += self.num_population_members

        # put the lowest energy into the best solution position.
        minval = np.argmin(self.population_energies)
        self._swap_best(minval)

        if warning_flag:
            return OptimizeResult(
                           x=self.x,
                           fun=self.population_energies[0],
                           nfev=nfev,
                           nit=nit,
                           message=status_message,
                           success=(warning_flag is not True))

        # do the optimisation.
        trials = np.zeros_like(self.population, order='F')
        for nit in range(1, self.maxiter + 1):
            if self.dither is not None:
                self.scale = self.random_number_generator.rand(
                ) * (self.dither[1] - self.dither[0]) + self.dither[0]

            # Unlike the standard DE, all the trials are created first and later
            # evaluated simultaneously.
            for index in range(self.num_population_members):
                # create a trial solution
                trials[index][:] = self._mutate(index)

                # ensuring that it's in the range [0, 1)
                self._ensure_constraint(trials[index])

                # scale from [0, 1) to the actual parameter value
                parameters[index][:] = self._scale_parameters(trials[index])

            # determine the energy of the objective function
            energies = self.evaluate_func(parameters)
            nfev += self.num_population_members

            # if the energy of the trial candidate is lower than the
            # original population member then replace it
            for index in range(self.num_population_members):
                if energies[index] < self.population_energies[index]:
                    self.population[index] = trials[index]
                    self.population_energies[index] = energies[index]

            # if the trial candidate also has a lower energy than the
            # best solution then replace that as well
            minval = np.argmin(self.population_energies)
            self._swap_best(minval)

            # stop when the fractional s.d. of the population is less than tol
            # of the mean energy
            convergence = (np.std(self.population_energies) /
                           np.abs(np.mean(self.population_energies) +
                                  _MACHEPS))

            if self.disp:
                print("differential_evolution step %d: f(x)= %g"
                      % (nit,
                         self.population_energies[0]))

            if self.callbacks:
                for callback in self.callbacks:
                    callback(step=nit, parameter=self.x,
                             cost=self.population_energies[0])

            if (self.earlystop and
                    self.earlystop(self.x,
                                  convergence=self.tol / convergence) is True):

                warning_flag = True
                status_message = ('earlystop function requested stop early '
                                  'by returning True')
                break

            if convergence < self.tol or warning_flag:
                break

        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True))

        if self.polish:
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method='L-BFGS-B',
                              bounds=self.limits.T,
                              args=self.args)

            nfev += result.nfev
            DE_result.nfev = nfev

            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        return DE_result

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
        for index, param in enumerate(trial):
            if param > 1 or param < 0:
                trial[index] = self.random_number_generator.rand()

    def _mutate(self, candidate):
        """
        create a trial vector based on a mutation strategy
        """
        trial = np.copy(self.population[candidate])

        rng = self.random_number_generator

        fill_point = rng.randint(0, self.parameter_count)

        if (self.strategy == 'randtobest1exp'
                or self.strategy == 'randtobest1bin'):
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = rng.rand(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            while (i < self.parameter_count and
                   rng.rand() < self.cross_over_probability):

                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, candidate, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r0, r1 = samples[:2]
        bprime = np.copy(self.population[candidate])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r0] -
                                self.population[r1])
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1]
                   - self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs

    def _swap_best(self, i):
        """
        put the i-th solution into the best solution position.
        """
        self.population_energies[[0, i]] = self.population_energies[[i, 0]]
        self.population[[0, i], :] = self.population[[i, 0], :]


def _make_random_gen(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
