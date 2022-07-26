"""For SERUMS Overbound Estimation."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp
import serums.models


def grimshaw_MLE(dataset):
    """Find estimates for shape and scale using Grimshaw's MLE Procedure.

    Inputs:
        dataset: 1D vector containing error data with 0 mean

    Outputs:
        2 element vector containing [shape_gamma, scale_beta]
    """
    # Calculate sample size, mean, minimum, and maximum
    n = dataset.size
    x_bar = np.mean(dataset)
    x_min = np.amin(dataset)
    x_max = np.amax(dataset)

    # GM Step 1: Choose epsilon
    epsilon = 1e-6 / x_bar

    # GM Step 2: Compute lower and upper bounds for zeros of h(theta)
    theta_L = (2 * (x_min - x_bar)) / (x_min**2)
    theta_U = (1 / x_max) - epsilon

    # GM Step 3: Compute lim of h''(theta) as theta goes to zero
    def GM3_series_term(x_i, x_bar):
        """Calculates value of single term in GM step 3 series."""
        return (x_i**2) - (2 * (x_bar**2))

    GM3_limit_test = (1 / n) * np.sum(GM3_series_term(dataset, x_bar))

    # GM Steps 4 and 5 (Find zeros of h(theta) if possible)

    def h(theta, data):
        """Returns value of h(theta) function for given theta and dataset."""
        value = (1 + (1 / data.size) * np.sum(np.log(1 - theta * data))) * (
            (1 / data.size) * np.sum(1 / (1 - theta * data))
        ) - 1
        return value

    def h_prime(theta, data):
        """Returns value of h'(theta) for a given theta and dataset."""
        value = (1 / theta) * (
            ((1 / data.size) * np.sum(1 / (1 - theta * data) ** 2))
            - ((1 / data.size) * np.sum(1 / (1 - theta * data))) ** 2
            - ((1 / data.size) * np.sum(np.log(1 - theta * data)))
            * (
                ((1 / data.size) * np.sum(1 / (1 - theta * data)))
                - ((1 / data.size) * np.sum(1 / (1 - theta * data) ** 2))
            )
        )
        return value

    def mod_NR_root_search(init_guess_theta, low, high, data):
        """Searches for a root using MNRM and returns the root if converged."""
        max_iter = 100000
        tol = 1e-15

        def mid(t1, t2):
            return (t1 + t2) / 2

        t_old = init_guess_theta
        bottom = low
        top = high

        for i in range(max_iter):
            t_sub = t_old - h(t_old, data) / h_prime(t_old, data)
            if t_sub > bottom and t_sub < top:
                t_old = t_sub
            else:
                t_old = mid(top, bottom)
            if h(bottom, dataset) * h(t_old, dataset) < 0:
                top = t_old
            else:
                bottom = t_old
            if abs(h(t_old, data)) < tol:  # changed condition
                conv = True
                break
            else:
                conv = False

        if conv is True:
            return t_old
        else:
            return t_old

    def mm_root_search(low, high, data):
        """Finds a root of h(theta) using the midpoint method on given interval."""
        max_iter = 100000
        tol = 1e-15
        bottom = low
        top = high

        for i in range(max_iter):
            t_old = bottom + top / 2
            if h(bottom, dataset) * h(t_old, dataset) < 0:
                top = t_old
            else:
                bottom = t_old
            if abs(h(t_old, data)) < tol:
                conv = True
                break
            else:
                conv = False

        if conv is True:
            return t_old
        else:
            return t_old

    throots = np.zeros(4, dtype=float)

    # Search for zeros of h(theta) on low interval
    throots[0] = mod_NR_root_search(theta_L, theta_L, -epsilon, dataset)

    if throots[0] > theta_L + epsilon and throots[0] < -epsilon * 2:
        root_conv1 = True
        lfs_test = h(theta_L, dataset) * h(throots[0] - epsilon, dataset)
        rts_test = h(-epsilon, dataset) * h(throots[0] + epsilon, dataset)
        if lfs_test < 0:
            throots[1] = mm_root_search(theta_L, throots[0] - epsilon, dataset)
        if rts_test < 0:
            throots[1] = mm_root_search(
                throots[0] + epsilon, -epsilon, dataset
            )
        else:
            throots[1] = -epsilon
    else:
        root_conv1 = False

    # Search for zeros of h(theta) on high interval
    throots[2] = mod_NR_root_search(theta_U, epsilon, theta_U, dataset)

    if throots[2] > 2 * epsilon and throots[2] < theta_U - epsilon:
        root_conv2 = True
        lfs_test = h(epsilon, dataset) * h(throots[2] - epsilon, dataset)
        rts_test = h(throots[2] + epsilon, dataset) * h(theta_U, dataset)
        if lfs_test < 0:
            throots[3] = mm_root_search(epsilon, throots[2] - epsilon, dataset)
        if rts_test < 0:
            throots[3] = mm_root_search(throots[2] + epsilon, theta_U, dataset)
        else:
            throots[3] = theta_U
    else:
        root_conv2 = False

    # GM Step 6: Compute candidate shape and scale parameters and
    # evaluate GPD log-likelihood
    def get_GPD_parameters(theta, data):
        """Calculates the shape and scale parameters of GPD from given Theta.

        Inputs:
            theta: particalar value of theta as defined by Grimshaw
            data: the entire set of error data

        Output:
            Tuple containing (shape, scale) using JDL convention (gamma/beta).
        """
        if abs(theta) < 1e-6:
            val1 = 1
            val2 = 100000000
        else:
            val1 = -(1 / data.size) * np.sum(np.log(1 - theta * dataset))
            val2 = val1 / theta

        tup = (-val1, val2)
        return list(tup)

    def GPD_log_likelihood(shape_gamma, scale_beta, data):
        """Evaluates the GPD log-likelihood given shape and scale."""
        k = -shape_gamma
        alfa = scale_beta
        if k == 0:
            val = -data.size * np.log(alfa) - (1 / alfa) * np.sum(data)
        else:
            val = -data.size * np.log(alfa) + ((1 / k) - 1) * np.sum(
                np.log(1 - (k * data) / alfa)
            )
        return val

    param_list = np.zeros([8], dtype=float)
    param_list[0:2] = get_GPD_parameters(throots[0], dataset)
    param_list[2:4] = get_GPD_parameters(throots[1], dataset)
    param_list[4:6] = get_GPD_parameters(throots[2], dataset)
    param_list[6:8] = get_GPD_parameters(throots[3], dataset)

    odds_list = np.zeros([4], dtype=float)
    odds_list[0] = GPD_log_likelihood(param_list[0], param_list[1], dataset)
    odds_list[1] = GPD_log_likelihood(param_list[2], param_list[3], dataset)
    odds_list[2] = GPD_log_likelihood(param_list[4], param_list[5], dataset)
    odds_list[3] = GPD_log_likelihood(param_list[6], param_list[7], dataset)

    idx = np.argmax(odds_list)
    crit = get_GPD_parameters(throots[idx], dataset)

    # GM Step 7 and 8: Choose best parameter estimates
    if root_conv1 is True or root_conv2 is True:
        if odds_list[idx] > -n * np.log(x_max):
            shape_gamma = crit[0]
            scale_beta = crit[1]
        else:
            shape_gamma = -1
            scale_beta = x_max
    else:
        shape_gamma = 0
        scale_beta = 0

    return np.array([shape_gamma, scale_beta], dtype=float)
