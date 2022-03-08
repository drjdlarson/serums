"""Estimates Parameters of Various Distributions."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp, cramervonmises_2samp
import serums.models


def _edparams_cost_factory(dist):
    def cost_function(x, rng, samples, method):
        # sample from distribution given by params in x
        if isinstance(dist, serums.models.Gaussian):
            x_samples = x[1] * rng.standard_normal(size=samples.size) + x[0]
        elif isinstance(dist, serums.models.StudentsT):
            x_samples = x[1] * rng.standard_t(np.abs(x[2]),
                                              size=samples.size) + x[0]
        elif isinstance(dist, serums.models.Cauchy):
            x_samples = x[1] * rng.standard_t(1, size=samples.size) + x[0]
        else:
            fmt = 'Invalid distribution choice: {}'
            raise RuntimeError(fmt.format(type(dist).__name__))

        # Kolmogorov-Smirnov test to get Chi2
        if method is serums.enums.GoodnessOfFitTest.KOLMOGOROV_SMIRNOV:
            cost = ks_2samp(x_samples, samples)[0]
        elif method is serums.enums.GoodnessOfFitTest.CRAMER_VON_MISES:
            out = cramervonmises_2samp(x_samples, samples)
            cost = out.statistic
        else:
            fmt = 'Invalid method choice: {}'
            raise RuntimeError(fmt.format(method))
        return cost
    return cost_function


def estimate_distribution_params(dist, method, samples,
                                 maxiter=100, disp=False):
    """Estimate distribution parameters from data.

    Parameters
    ----------
    dist : :class:`serums.models.BaseSingleModel`
        Desired target distribution to fit samples.
    method : :class:`serums.enums.GoodnessOfFitTest`
        Selection of goodness of fit test to compare samples to target distribution.
    samples : N numpy array
        Samples to be fitted to the input distribution.
    maxiter : int, optional
        Maximum number of iterations to run the estimator. The default is 100.
    disp : bool, optional
        Flag for displaying extra information. The default is False.

    Raises
    ------
    RuntimeError
        Raised if the estimator fails.

    Returns
    -------
    output : :class:`serums.models.BaseSingleModel`
        Estimated output distribution.
    """
    output = {}

    # Select cost function
    costfun = _edparams_cost_factory(dist)
    if isinstance(dist, (serums.models.Gaussian, serums.models.Cauchy)):
        x0 = np.array([dist.location.item(), dist.scale.item()])
        rng = np.random.default_rng()
    elif isinstance(dist, serums.models.StudentsT):
        x0 = np.array([dist.location.item(), dist.scale.item(),
                       dist.degrees_of_freedom.item()])
        rng = np.random.default_rng()
    else:
        fmt = 'Invalid distribution choice: {}'
        raise RuntimeError(fmt.format(type(dist).__name__))

    # optimize distribution parameters
    res = minimize(costfun, x0, args=(rng, samples, method),
                   method='Powell',
                   options={'maxiter': maxiter, 'disp': disp})

    if not res.success:
        fmt = 'Parameter estimation failed with:\n{}'
        raise RuntimeError(fmt.format(res.message))

    # Set outputs
    output = type(dist)()
    #temporary inclusion of the 1 in reshape, need to modify this later on potentially.
    output.location = res.x[0].reshape(np.shape(dist.location))
    output.scale = np.abs(res.x[1]).reshape(np.shape(dist.scale))

    if isinstance(dist, serums.models.StudentsT):
        output.degrees_of_freedom = np.abs(res.x[2]).item()
    elif isinstance(dist, (serums.models.Gaussian, serums.models.Cauchy)):
        pass
    else:
        fmt = 'Invalid distribution choice: {}'
        raise RuntimeError(fmt.format(type(dist).__name__))

    return output
