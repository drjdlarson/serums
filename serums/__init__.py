"""Package for Statistical Error and Risk Utility for Multi-sensor Systems."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp, cramervonmises_2samp


def _edparams_cost_factory(dist_type):
    def cost_function(x, rng, samples, method):
        # sample from distribution given by params in x
        if dist_type.lower() == 'gaussian':
            x_samples = x[1] * rng.standard_normal(size=samples.size) + x[0]
        elif dist_type.lower() == 'studentst':
            x_samples = x[1] * rng.standard_t(np.abs(x[2]),
                                              size=samples.size) + x[0]
        elif dist_type.lower() == 'cauchy':
            x_samples = x[1] * rng.standard_t(1, size=samples.size) + x[0]
        else:
            fmt = 'Invalid distribution choice: {}'
            raise RuntimeError(fmt.format(dist_type))

        # Kolmogorov-Smirnov test to get Chi2
        if method.lower() == 'kolmogorovsmirnov':
            cost = ks_2samp(x_samples, samples)[0]
        elif method.lower() == 'cramervonmises':
            out = cramervonmises_2samp(x_samples, samples)
            cost = out.statistic
        else:
            fmt = 'Invalid method choice: {}'
            raise RuntimeError(fmt.format(method))
        return cost
    return cost_function


def estimate_distribution_params(params, dist_type, method, samples,
                                 maxiter=100, disp=False):
    """Estimate distribution parameters from data.

    Parameters
    ----------
    params : dict
        DESCRIPTION.
    dist_type : TYPE
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.
    args : TYPE
        DESCRIPTION.
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
    output : dict
        DESCRIPTION.
    """
    output = {}

    # Select cost function
    costfun = _edparams_cost_factory(dist_type)
    if dist_type.lower() in ('gaussian', 'cauchy'):
        x0 = np.array([params['loc'], params['scale']])
        rng = np.random.default_rng()
    elif dist_type.lower() == 'studentst':
        x0 = np.array([params['loc'], params['scale'], params['df']])
        rng = np.random.default_rng()
    else:
        fmt = 'Invalid distribution choice: {}'
        raise RuntimeError(fmt.format(dist_type))

    # optimize distribution parameters
    res = minimize(costfun, x0, args=(rng, samples, method),
                   method='Powell',
                   options={'maxiter': maxiter, 'disp': disp})

    if not res.success:
        fmt = 'Parameter estimation failed with:\n{}'
        raise RuntimeError(fmt.format(res.message))

    # Set outputs
    if dist_type.lower() in ('gaussian', 'cauchy'):
        output['loc'] = res.x[0]
        output['scale'] = np.abs(res.x[1])
    elif dist_type.lower() == 'studentst':
        output['loc'] = res.x[0]
        output['scale'] = np.abs(res.x[1])
        output['df'] = np.abs(res.x[2])
    else:
        fmt = 'Invalid distribution choice: {}'
        raise RuntimeError(fmt.format(dist_type))

    return output
