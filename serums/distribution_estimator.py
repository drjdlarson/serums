"""Estimates Parameters of Various Distributions."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp
import serums.models
import serums.enums as senums
import serums.errors


def _edparams_cost_factory(dist):
    def cost_function(x, rng, samples, method):
        # sample from distribution given by params in x
        if isinstance(dist, serums.models.Gaussian):
            x_samples = x[1] * rng.standard_normal(size=samples.size) + x[0]

        elif isinstance(dist, serums.models.Cauchy):
            x_samples = x[1] * rng.standard_t(1, size=samples.size) + x[0]

        elif isinstance(dist, serums.models.StudentsT):
            x_samples = (
                x[1] * rng.standard_t(np.abs(x[2]), size=samples.size) + x[0]
            )
        else:
            fmt = "Invalid distribution choice: {}"
            raise RuntimeError(fmt.format(type(dist).__name__))

        # Kolmogorov-Smirnov test to get Chi2
        if method is senums.DistEstimatorMethod.KOLMOGOROV_SMIRNOV:
            cost = ks_2samp(x_samples, samples)[0]
        elif method is senums.DistEstimatorMethod.CRAMER_VON_MISES:
            out = cramervonmises_2samp(x_samples, samples)
            cost = out.statistic
        elif method is senums.DistEstimatorMethod.ANDERSON_DARLING:
            cost = anderson_ksamp([x_samples, samples]).statistic
        else:
            fmt = "Invalid method choice: {}"
            raise RuntimeError(fmt.format(method))
        return cost

    return cost_function


def estimate_distribution_params(
    dist, method, samples, maxiter=100, disp=False
):
    """Estimate distribution parameters from data.

    Parameters
    ----------
    dist : :class:`serums.models.BaseSingleModel`
        Desired target distribution to fit samples.
    method : :class:`serums.enums.DistEstimatorMethod`
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
    # Set outputs
    output = type(dist)()

    if method in (
        senums.DistEstimatorMethod.ANDERSON_DARLING,
        senums.DistEstimatorMethod.CRAMER_VON_MISES,
        senums.DistEstimatorMethod.KOLMOGOROV_SMIRNOV,
    ):
        # Select cost function
        costfun = _edparams_cost_factory(dist)
        if isinstance(dist, (serums.models.Gaussian, serums.models.Cauchy)):
            x0 = np.array([dist.location.item(), dist.scale.item()])
            rng = np.random.default_rng()
        elif isinstance(dist, serums.models.StudentsT):
            x0 = np.array(
                [
                    dist.location.item(),
                    dist.scale.item(),
                    dist.degrees_of_freedom.item(),
                ]
            )
            rng = np.random.default_rng()
        else:
            fmt = "Invalid distribution choice: {}"
            raise RuntimeError(fmt.format(type(dist).__name__))

        # optimize distribution parameters
        res = minimize(
            costfun,
            x0,
            args=(rng, samples, method),
            method="Powell",
            options={"maxiter": maxiter, "disp": disp},
        )

        if not res.success:
            fmt = "Parameter estimation failed with:\n{}"
            raise RuntimeError(fmt.format(res.message))

        output.location = res.x[0].reshape(np.shape(dist.location))
        output.scale = np.abs(res.x[1]).reshape(np.shape(dist.scale))

        if isinstance(dist, (serums.models.Gaussian, serums.models.Cauchy)):
            pass

        elif isinstance(dist, serums.models.StudentsT):
            output.degrees_of_freedom = np.abs(res.x[2]).item()

        else:
            fmt = "Invalid distribution choice: {}"
            raise RuntimeError(fmt.format(type(dist).__name__))

    elif method is senums.DistEstimatorMethod.GRIMSHAW_MLE:
        if not isinstance(dist, serums.models.GeneralizedPareto):
            raise RuntimeError("Invalid distribution for Grimshaws MLE")
        try:
            shape, scale = grimshaw_MLE(samples)[:2]
            output.shape = np.array([[shape]])
            output.scale = np.array([[scale]])
        except serums.errors.DistributionEstimatorFailed:
            estimate_distribution_params(
                dist,
                senums.DistEstimatorMethod.METHOD_OF_MOMENTS,
                samples,
                maxiter=maxiter,
                disp=disp,
            )

    elif method is senums.DistEstimatorMethod.METHOD_OF_MOMENTS:
        if not isinstance(dist, serums.models.GeneralizedPareto):
            raise RuntimeError("Method of Moments only implemented for GPD")
        else:
            shape, scale = genpareto_MOM_est(samples)[:2]
            output.shape = np.array([[shape]])
            output.scale = np.array([[scale]])

    elif method is senums.DistEstimatorMethod.PROBABILITY_WEIGHTED_MOMENTS:
        if not isinstance(dist, serums.models.GeneralizedPareto):
            raise RuntimeError("PWM only implemented for GPD")
        else:
            shape, scale = genpareto_PWM_est(samples)[:2]
            output.shape = np.array([[shape]])
            output.scale = np.array([[scale]])

    return output


def grimshaw_MLE(dataset):
    """Finds estimates for GPD shape and scale using Grimshaw's MLE Procedure.

    Notes
    -----
    This method is taken from
    :cite:`Grimshaw1993_ComputingMLEforGeneralizedParetoDistribution`

    Parameters
    ----------
    dataset : N numpy array
        Contains positive error data above a threshold of 0.

    Returns
    -------
    shape_gamma : float
        Shape parameter as defined in chapter 3.2 of
        :cite:'LarsonThesis'
    scale_beta : float
        Scale parameter as defined in chapter 3.2 of
        :cite:'LarsonThesis'
        Contains the estimated covariance matrix for shape, scale based on
        observed Fisher information. For further information, see chapter 4.1
        of
        :cite:'LarsonThesis'
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

    # Bypass GM Step 3: Search for zeros on both intervals regardless of h''(t)

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
        max_iter = 1000
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
            elif t_old == low or t_old == high:
                conv = False
                break
            else:
                conv = False

        if conv is True:
            return t_old
        else:
            return t_old

    def mm_root_search(low, high, data):
        """Finds a root of h(theta) using the midpoint method on given interval."""
        max_iter = 1000
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
        """Calculates shape, scale of GPD from given Theta."""
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
        raise serums.errors.DistributionEstimatorFailed("Grimshaw MLE failed")

    # Calculate covariance of shape, scale parameters using CRLB based on
    # observed Fisher information (eq. 4.2 in Larson dissertation)

    eye_11 = (
        (2 / (shape_gamma**3))
        * np.sum(np.log(1 + (shape_gamma / scale_beta) * dataset))
        - (2 / (shape_gamma**2))
        * np.sum(
            (dataset / scale_beta) / (1 + (shape_gamma / scale_beta) * dataset)
        )
        - (1 + (1 / shape_gamma))
        * np.sum(
            (
                (dataset / scale_beta)
                / (1 + (shape_gamma / scale_beta) * dataset)
            )
            ** 2
        )
    )
    eye_12 = -np.sum(
        (dataset / scale_beta**2)
        / (1 + (shape_gamma / scale_beta) * dataset)
    ) + (1 + shape_gamma) * np.sum(
        (dataset**2 / scale_beta**3)
        / (1 + (shape_gamma / scale_beta) * dataset) ** 2
    )
    eye_21 = eye_12
    eye_22 = (
        (-n / scale_beta**2)
        + 2
        * (shape_gamma + 1)
        * np.sum(
            (dataset / scale_beta**3)
            / (1 + (shape_gamma / scale_beta) * dataset)
        )
        - shape_gamma
        * (shape_gamma + 1)
        * np.sum(
            (dataset**2 / scale_beta**4)
            / (1 + (shape_gamma / scale_beta) * dataset) ** 2
        )
    )

    sub = np.array([[eye_11, eye_12], [eye_21, eye_22]])
    gpd_covar = np.linalg.inv(sub)

    return shape_gamma, scale_beta, gpd_covar


def genpareto_MOM_est(dataset):
    """Finds estimates for GPD shape and scale using the method of moments.

    Important: The covariance of these estimators is only shown to be
    asymptotically normally distributed and given by the formula below if
    (shape_gamma < 1/4) is true

    Notes
    -----
    This method is taken from
    :cite:`HoskingWallis1987_ParameterEstimationforGeneralizedParetoDistribution`

    Parameters
    ----------
    dataset : N x 1 numpy array
        Contains positive error data above threshold

    Returns
    -------
    shape_gamma : float
        Shape parameter as defined by Larson dissertation
    scale_beta : float
        Scale parameter as defined by Larson dissertation
    gpd_covar : 2 x 2 numpy array
        Contains the estimated asymptotic covariance matrix for shape, scale
        given in Hosking and Wallis
    """
    x_bar = np.mean(dataset)
    svar = np.var(dataset, ddof=1)

    alfahat = 0.5 * x_bar * (1 + (x_bar**2) / svar)
    khat = 0.5 * (-1 + (x_bar**2) / svar)

    shape_gamma = -khat
    scale_beta = alfahat

    # Calculate estimated asymptotic covariance matrix
    n = dataset.size
    gain = (1 / n) * (
        ((1 + khat) ** 2) / ((1 + 2 * khat) * (1 + 3 * khat) * (1 + 4 * khat))
    )

    mat11 = 2 * (alfahat**2) * (1 + 6 * khat + 12 * khat**2)
    mat12 = alfahat * (1 + 2 * khat) * (1 + 4 * khat + 12 * khat**2)
    mat21 = mat12
    mat22 = ((1 + 2 * khat) ** 2) * (1 + khat + 6 * khat**2)

    gpd_covar = gain * np.array([[mat11, mat12], [mat21, mat22]])

    return shape_gamma, scale_beta, gpd_covar


def genpareto_PWM_est(dataset):
    """Finds estimates for GPD shape and scale using the PWM method.

    Important: The covariance matrix calculated below is only valid if the
    shape parameter gamma is less than 0.5

    Notes
    -----
    This method is taken from
    :cite:`HoskingWallis1987_ParameterEstimationforGeneralizedParetoDistribution`

    Parameters
    ----------
    dataset : N x 1 numpy array
        Contains positive error data above threshold

    Returns
    -------
    shape_gamma : float
        Shape parameter as defined by Larson dissertation
    scale_beta : float
        Scale parameter as defined by Larson dissertation
    gpd_covar : 2 x 2 numpy array
        Contains the estimated asymptotic covariance matrix for shape, scale
        given in Hosking and Wallis
    """
    n = dataset.size
    ordered_list = np.sort(dataset)

    gam0bar = (1 / n) * np.sum(dataset)

    summate = 0
    for ii in range(n):
        temp = ((n - (ii + 1)) / (n - 1)) * ordered_list[ii]
        summate = summate + temp

    gam1bar = (1 / n) * summate

    scale_beta = (2 * gam0bar * gam1bar) / (gam0bar - 2 * gam1bar)
    shape_gamma = -(gam0bar / (gam0bar - 2 * gam1bar) - 2)

    # Calculate estimated asymptotic covariance matrix
    k = -shape_gamma
    a = scale_beta

    gain = (1 / n) * (1 / ((1 + 2 * k) * (3 + 2 * k)))

    mat11 = (a**2) * (7 + 18 * k + 11 * k**2 + 2 * k**3)
    mat12 = a * (2 + k) * (2 + 6 * k + 7 * k**2 + 2 * k**3)
    mat21 = mat12
    mat22 = (1 + k) * ((2 + k) ** 2) * (1 + k + 2 * k**2)

    gpd_covar = gain * np.array([[mat11, mat12], [mat21, mat22]])

    return shape_gamma, scale_beta, gpd_covar
