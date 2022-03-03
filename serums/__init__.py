"""Package for Statistical Error and Risk Utility for Multi-sensor Systems."""
import numpy as np
from scipy.optimize import minimize


# TODO: placeholder for now, must have this signature. x is 1-d array of params to est, *args are other data probably data set
def _gaussian_cost(x, *args):
    # sample from distribution given by params in x

    # Kolmogorov-Smirnov test to get Chi2

    # return -Chi2
    pass


def estimate_distribution_params(params, dist_type, method, args,
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
    if dist_type.lower() == 'gaussian':
        x0 = np.array([params['mean'], params['std']])
        res = minimize(_gaussian_cost, x0, args=args, method='Powell',
                       options={'maxiter': maxiter, 'disp': disp})
        if not res.success:
            fmt = 'Parameter estimation failed with:\n{}'
            raise RuntimeError(fmt.format(res.message))

        output['mean'] = res.x[0]
        output['std'] = res.x[1]

    return output
