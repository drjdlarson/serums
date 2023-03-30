"""For SERUMS Overbound Estimation."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, basinhopping
import serums.models as smodels
import serums.distribution_estimator as de
from abc import abstractmethod, ABC
from scipy.stats import norm, halfnorm, t
import serums.errors
import scipy.special as special
from typing import List, Callable


def fusion(
    varList: List[smodels.BaseSingleModel | smodels.BaseMixtureModel],
    poly: Callable[
        [List[smodels.BaseSingleModel | smodels.BaseMixtureModel]], np.ndarray
    ],
) -> np.ndarray:
    max_size = max([x.monte_carlo_size for x in varList])
    for x in varList:
        x.monte_carlo_size = int(max_size)
    return poly(*varList)


class OverbounderBase(ABC):
    """Represents base class for any overbound object."""

    def __init__(self):
        pass

    def erf_gauss(self, X, mean, sigma):
        """Obtain numerical approximation of erf inside Gaussian CDF equation."""
        phi = (X - mean) / (2**0.5)
        n_terms = 55
        gain = 2 / (np.pi**0.5)

        terms = np.zeros(n_terms)
        n = 0

        while n < n_terms:
            terms[n] = (
                ((-1) ** n)
                * (1 / (2 * n + 1))
                * (1 / np.math.factorial(n))
                * (phi / sigma) ** (2 * n + 1)
            )
            n = n + 1

        return gain * sum(terms)

    def dsig_erf_gauss(self, X, mean, sigma):
        """Obtain numerical approximation of d/dsigma{erf} for Newtons method."""
        phi = (X - mean) / (2**0.5)
        n_terms = 55
        gain = 2 / (np.pi**0.5)

        terms = np.zeros(n_terms)
        n = 0

        while n < n_terms:
            terms[n] = (
                ((-1) ** (n + 1))
                * (1 / np.math.factorial(n))
                * (phi ** (2 * n + 1))
                * ((1 / sigma) ** (2 * n + 2))
            )
            n = n + 1

        return gain * sum(terms)

    def g_sigma(self, mean, Xi, Yi, sigma):
        """Determine value of g(sigma) given Gauss mean and point on Gauss CDF."""
        sub = 1 - 2 * Yi
        return sub + self.erf_gauss(Xi, mean, sigma)

    def g_prime_sigma(self, mean, Xi, Yi, sigma):
        """Determine value of g'(sigma) given Gauss mean and point on Gauss CDF."""
        return self.dsig_erf_gauss(Xi, mean, sigma)

    def h_sigma(self, Xi, Yi, sigma):
        """Determine value of h(sigma) Given point on half-gaussian ECDF."""
        return -Yi + self.erf_gauss(Xi, 0, sigma)

    def h_prime_sigma(self, Xi, Yi, sigma):
        """Determine value of h'(sigma) given point on half-gaussian ECDF."""
        return self.dsig_erf_gauss(Xi, 0, sigma)

    def find_sigma_g(self, mean, Xi, Yi):
        """Find Gaussian st. dev. that intersects a certain mean and point on CDF."""
        sigma_0 = 0.75 * abs(Xi - mean)
        sigma_iter = sigma_0
        i = 1

        while abs(self.g_sigma(mean, Xi, Yi, sigma_iter)) > 1e-14:
            sigma_iter = sigma_iter - (
                self.g_sigma(mean, Xi, Yi, sigma_iter)
                / self.g_prime_sigma(mean, Xi, Yi, sigma_iter)
            )
            i = i + 1

        return sigma_iter

    def find_sigma_h(self, Xi, Yi):
        """Find half-Gauss sigma that intersects a certain point on CDF."""
        sigma_0 = 0.75 * abs(Xi)
        sigma_iter = sigma_0
        i = 1

        while abs(self.h_sigma(Xi, Yi, sigma_iter)) > 1e-14:
            sigma_iter = sigma_iter - (
                self.h_sigma(Xi, Yi, sigma_iter)
                / self.h_prime_sigma(Xi, Yi, sigma_iter)
            )
            i = i + 1
        return sigma_iter

    def get_pareto_scale(self, Xi, Yi, shape):
        """Find GPD scale given shape and point on CDF."""
        return (shape * Xi) / (((1 - Yi) ** -shape) - 1)

    def gauss_1pt_pierce_cost(self, sigma, pt):
        """Cost function to be optimized when finding piercing half Gaussian.

        Parameters
        ----------
        sigma : 1 numpy array
            Half-Gaussian standard deviation to be searched over, i.e. the
            independent variable which is iterated during the optimization
            routine.
        pt : 2x1 numpy array
            Point to seek Half-Gaussian pierce with (0,0). Format is (x, y)

        Returns
        -------
        error : float
            Magnitude of the error in the dependent variable between the actual
            point and the Half-Gaussian CDF value at the point.

        """
        error = np.abs(pt[1] - halfnorm.cdf(pt[0], loc=0, scale=sigma))
        return error

    def gauss_2pt_pierce_cost(self, gauss_param_array, pts):
        """Cost function for optimization of Gaussian pierce search.

        Parameters
        ----------
        gauss_param_array : 2 numpy array
            Gauss parameters to be searched over. Format is (mu, sigma).
        pts : 2x2 numpy array
            Points to seek pierce between. Row 1 is (x1, x2). Row 2 is (y1, y2).

        Returns
        -------
        norm : float
            L2 norm of vector containing errors in CDF function at points 1 & 2.

        """
        vect = -pts[1] + 0.5 * (
            1
            + special.erf(
                (pts[0] - gauss_param_array[0])
                / (np.sqrt(2) * gauss_param_array[1])
            )
        )

        return np.sum(vect**2)

    @abstractmethod
    def overbound(self, *args, **kwargs):
        """Produce the overbound of the type denoted by the child class name."""
        raise NotImplementedError("not implemented")


class SymmetricGaussianOverbounder(OverbounderBase):
    """Represents a Symmetric Gaussian Overbounder object."""

    def __init__(self):
        super().__init__()
        self.pierce_locator = 1

    def overbound(self, data):
        """Produce Gaussian model object that overbounds given error data."""
        n = data.size
        sample_sigma = np.std(data, ddof=1)
        threshold = self.pierce_locator * sample_sigma
        OB_mean = 0

        # Determine points on Half-Gaussian ECDF
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        sub = np.absolute(data)
        abs_data = np.sort(sub)

        # Determine points on lower DKW band
        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        DKW_low = np.subtract(ecdf_ords, epsilon)

        # Determine points for intersection check
        sub = np.array(np.where(abs_data > threshold))
        candidates = np.zeros(n)

        # Determine list of candidate sigmas
        for i in sub[0, 0:]:
            pt = np.array([[abs_data[i]], [DKW_low[i]]])
            ans = minimize(
                self.gauss_1pt_pierce_cost,
                sample_sigma,
                args=(pt,),
                method="Powell",
                options={"xtol": 1e-14, "maxfev": 10000, "maxiter": 10000},
            )
            if ans.success is True:
                candidates[i] = ans.x[0]
            else:
                candidates[i] = 0

        # Select maximum sigma with irrationality protection (likely unneccessary
        # if disregarding inner core)
        rational_candidates = candidates[~np.isnan(candidates)]
        OB_sigma = np.max(rational_candidates)

        return smodels.Gaussian(
            mean=np.array([[OB_mean]]), covariance=np.array([[OB_sigma**2]])
        )


class SymmetricGPO(OverbounderBase):
    """Represents a Symmetric Gaussian-Pareto Overbounder object."""

    def __init__(self):
        super().__init__()
        self.inside_pierce_locator = 1
        self.ThresholdReductionFactor = 1

    def overbound(self, data):
        """Produce symmetric Gaussian-Pareto mixture model object that overbounds input error data."""
        n = data.size
        sigma = np.sqrt(np.var(data, ddof=1))
        print("\nComputing Symmetric Gaussian-Pareto Overbound...")
        # print(
        #     "Sample St. Dev. : ",
        #     sigma,
        # )

        pos = np.absolute(data)
        sorted_abs_data = np.sort(pos)
        # print("Most Extreme Deviation in Sample: ", sorted_abs_data[-1])

        Nt_min = 250
        Nt_max = int(np.ceil(0.1 * n))
        idx_u_min = n - Nt_max - 1
        idx_u_max = n - Nt_min - 1
        u_idxs = np.arange(idx_u_min, idx_u_max + 1)
        u_candidates = sorted_abs_data[u_idxs]
        Nu = u_candidates.size
        shapes = np.zeros(Nu)
        MLE_used = np.zeros(Nu, dtype=bool)

        print("\nComputing Tail GPD Shape Parameter Estimates...\nProgress:")

        resolution = 10 * (1 / Nu)
        for i in range(Nu):
            if i < Nu - 1:
                fraction = i / Nu
                checkpoint = np.floor(10 * fraction)
                diagnostic = (10 * fraction) - checkpoint
                if diagnostic < resolution:
                    print("{:3d}%".format(int(checkpoint * 10)))
            else:
                print("100%")

            try:
                shapes[i] = de.grimshaw_MLE(
                    np.subtract(
                        sorted_abs_data[u_idxs[i] + 1 :],
                        sorted_abs_data[u_idxs[i]] - 1e-25,
                    )
                )[0]
                MLE_used[i] = True
            except serums.errors.DistributionEstimatorFailed:
                shapes[i] = 0
            # print(shapes[i], "\n", MLE_used[i])

        shape_max = max(shapes)
        idx_shape_max = np.where(shapes == shape_max)[0]
        if shape_max <= 0:
            raise serums.errors.OverboundingMethodFailed(
                "MLE indicates exponential or finite tail. Use the Symmetric Gaussian Overbounder."
            )

        shape_max_covar = de.grimshaw_MLE(
            np.subtract(
                sorted_abs_data[u_idxs[idx_shape_max[0]] + 1 :],
                sorted_abs_data[u_idxs[idx_shape_max[0]]] - 1e-25,
            )
        )[2]

        tail_size = (n - 1) - u_idxs[idx_shape_max[0]]
        tail_shape = shape_max + t.ppf(0.975, tail_size) * np.sqrt(
            shape_max_covar[1, 1]
        )
        # print("Maximum Tail GPD Shape Parameter Estimate : ", tail_shape)
        u_idx = u_idxs[idx_shape_max[0]]
        u = sorted_abs_data[u_idx]
        # print("Corresponding Error Threshold: ", u)

        # Perform gaussian overbound on core region between inside locator and threshold
        print("\nComputing Gaussian Overbound for Core Region...")

        # Determine points on Half-Gaussian ECDF
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        # Determine points on lower DKW band
        # confidence = 1 - 1e-6
        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        DKW_low = np.subtract(ecdf_ords, epsilon)

        # Determine points for intersection check
        core_min = self.inside_pierce_locator * np.sqrt(
            np.var(sorted_abs_data[np.where(sorted_abs_data < u)[0]], ddof=1)
        )
        sub = np.where(sorted_abs_data > core_min)[0]
        start = sub[0]
        sub = np.where(sorted_abs_data < u)[0]
        end = sub[-1]
        candidates = np.zeros(n)
        subs = np.arange(start, end + 2, 1)
        # print("Number of Points in Core Search Region: ", subs.size)
        # print(
        #     "Range of Errors Corresponding to Core Search Region: ",
        #     sorted_abs_data[start],
        #     "-",
        #     sorted_abs_data[end + 1],
        # )

        # Determine list of candidate sigmas
        for i in subs:
            candidates[i] = self.find_sigma_h(sorted_abs_data[i], DKW_low[i])

        # Select maximum sigma with NaN protection (likely unneccessary
        # if disregarding inner core)
        real_candidates = candidates[~np.isnan(candidates)]
        core_sigma = np.max(real_candidates)
        # print("St. Dev. of Gaussian Core Overbound:", core_sigma)
        # print(
        #     "Error Value Corresponding to Core CDF Pierce Condition: ",
        #     sorted_abs_data[np.where(candidates == core_sigma)[0]][0],
        # )

        # Determine tail scale parameter by checking for maximum scale
        # corresponding to pierce from threshold to end point in tail
        print("\nComputing Tail GPD Scale Parameter...")

        tail_idxs = np.arange(u_idx + 1, n, 1)
        tail_pts = sorted_abs_data[tail_idxs]
        shifted_tail_pts = np.zeros(n)
        shifted_tail_pts[tail_idxs] = np.subtract(
            sorted_abs_data[tail_idxs],
            sorted_abs_data[u_idx],
        )
        scales = np.zeros(n)

        # Transform lower DKW band to CEDF domain of the tail
        Fu = self.erf_gauss(u, 0, core_sigma)
        tail_DKW_ords_CEDF_domain = np.zeros(n)
        tail_DKW_ords_CEDF_domain[tail_idxs] = np.subtract(
            DKW_low[tail_idxs], Fu
        ) / (1 - Fu)

        for i in tail_idxs:
            scales[i] = self.get_pareto_scale(
                shifted_tail_pts[i], tail_DKW_ords_CEDF_domain[i], tail_shape
            )

        # Select maximum scale with irrationality protection
        rational_scales = scales[~np.isnan(scales)]
        tail_scale = np.max(rational_scales)
        # print(
        #     "Required Tail GPD Scale Parameter to Enforce Overbound:",
        #     tail_scale,
        # )
        print("\nDone.")
        return smodels.SymmetricGaussianPareto(
            scale=core_sigma,
            threshold=u,
            tail_shape=tail_shape,
            tail_scale=tail_scale,
        )


class PairedGaussianOverbounder(OverbounderBase):
    """Represents a Paired Gaussian Overbounder object."""

    def __init__(self):
        super().__init__()

    def _cost_left(self, params, x_check, y_check):
        y_curve = norm.cdf(x_check, loc=params[0], scale=params[1])
        cost_vect = y_curve - y_check
        pos_indices = cost_vect >= 0
        cost = np.sum(cost_vect[pos_indices])
        cost += np.sum(
            -1000 * y_check.size * cost_vect[np.logical_not(pos_indices)]
        )
        return cost

    def _cost_right(self, params, x_check, y_check):
        y_curve = norm.cdf(x_check, loc=params[0], scale=params[1])
        cost_vect = y_check - y_curve
        pos_indices = cost_vect >= 0
        cost = np.sum(cost_vect[pos_indices])
        cost += np.sum(
            -1000 * y_check.size * cost_vect[np.logical_not(pos_indices)]
        )
        return cost

    def overbound(self, data, debug_plots=False):
        """Produce Paired Gaussian model object that overbounds input error data.

        Parameters
        ----------
        data : N numpy array of error data

        Returns
        -------
        out_dist : :class:
            'PairedGaussian' object from serums/models.py
        """
        n = data.size
        sorted_data = np.sort(data)
        init_mean = np.mean(data)
        init_sigma = np.std(data, ddof=1)
        init_guess = np.array([init_mean, init_sigma])

        # Generate sample ECDF ordinates
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        # Compute Upper and Lower 95% DKW Confidence Bounds
        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        DKW_low = np.subtract(ecdf_ords, epsilon)
        DKW_high = np.add(ecdf_ords, epsilon)

        left_usable_idxs = np.asarray(DKW_high < (1 - epsilon)).nonzero()[0]
        x_check_left = sorted_data[left_usable_idxs]
        y_check_left = DKW_high[left_usable_idxs]

        left_result = basinhopping(
            self._cost_left,
            init_guess,
            niter=200,
            stepsize=np.array([init_mean / 2, init_sigma / 2]),
            minimizer_kwargs={
                "args": (x_check_left, y_check_left),
                "method": "Powell",
                "options": {
                    "xtol": 1e-14,
                    "ftol": 1e-6,
                    "maxfev": 10000,
                    "maxiter": 10000,
                },
            },
        )
        # if debug_plots:
        #     plt.figure("Test Plot Left")
        #     plt.plot(
        #         data[left_usable_idxs],
        #         DKW_high[left_usable_idxs],
        #         label="Allowable DKW High",
        #     )
        #     y_left_ob = norm.cdf(
        #         data, loc=left_result.x[0], scale=left_result.x[1]
        #     )
        #     plt.plot(
        #         data[left_usable_idxs],
        #         y_left_ob[left_usable_idxs],
        #         label="Left OB",
        #     )
        #     plt.legend()

        right_usable_idxs = np.asarray(DKW_low > epsilon).nonzero()[0]
        x_check_right = sorted_data[right_usable_idxs]
        y_check_right = DKW_low[right_usable_idxs]

        right_result = basinhopping(
            self._cost_right,
            init_guess,
            niter=200,
            stepsize=np.array([init_mean / 2, init_sigma / 2]),
            minimizer_kwargs={
                "args": (x_check_right, y_check_right),
                "method": "Powell",
                "options": {
                    "xtol": 1e-14,
                    "ftol": 1e-6,
                    "maxfev": 10000,
                    "maxiter": 10000,
                },
            },
        )
        # if debug_plots:
        #     plt.figure("Test Plot Right")
        #     plt.plot(
        #         data[right_usable_idxs],
        #         DKW_low[right_usable_idxs],
        #         label="Allowable DKW Low",
        #     )
        #     y_left_ob = norm.cdf(
        #         data, loc=right_result.x[0], scale=right_result.x[1]
        #     )
        #     plt.plot(
        #         data[right_usable_idxs],
        #         y_left_ob[right_usable_idxs],
        #         label="Right OB",
        #     )
        #     plt.legend()

        left_ob = smodels.Gaussian(
            mean=left_result.x[0],
            covariance=np.array([[left_result.x[1] ** 2]]),
        )
        right_ob = smodels.Gaussian(
            mean=right_result.x[0],
            covariance=np.array([[right_result.x[1] ** 2]]),
        )

        return smodels.PairedGaussian(left_ob, right_ob)


class PairedGPO(OverbounderBase):
    """Represents a Paired Gaussian-Pareto Overbounder Object."""

    def __init__(self):
        super().__init__()
        self.ThresholdReductionFactor = 1
        self.StrictPairedEnforcement = False

    def _cost_left(self, params, x_check, y_check):
        y_curve = norm.cdf(x_check, loc=params[0], scale=params[1])
        cost_vect = y_curve - y_check
        pos_indices = cost_vect >= 0
        cost = np.sum(cost_vect[pos_indices])
        cost += np.sum(
            -10000 * y_check.size * cost_vect[np.logical_not(pos_indices)]
        )
        return cost

    def _cost_right(self, params, x_check, y_check):
        y_curve = norm.cdf(x_check, loc=params[0], scale=params[1])
        cost_vect = y_check - y_curve
        pos_indices = cost_vect >= 0
        cost = np.sum(cost_vect[pos_indices])
        cost += np.sum(
            -10000 * y_check.size * cost_vect[np.logical_not(pos_indices)]
        )
        return cost

    def overbound(self, data):
        """Produce Paired Gaussian-Pareto model object that overbounds input error data.

        Parameters
        ----------
        data : N numpy array of error data

        Returns
        -------
        out_dist : :class:
            'PairedGaussianPareto' object from serums/models.py
        """
        n = data.size
        data_sorted = np.sort(data)
        idx_10p = int(np.ceil(0.1 * n))
        idxs_cand_u_left = np.arange(250, idx_10p, 1)

        max_shape_left = 0
        idx_u_left = None
        max_shape_covar_left = None
        for i in idxs_cand_u_left:
            try:
                shape, scale, covar = de.grimshaw_MLE(
                    np.add(
                        np.abs(
                            np.subtract(
                                data_sorted[0:i],
                                data_sorted[i],
                            )
                        ),
                        1e-14,
                    )
                )
                if shape > max_shape_left:
                    max_shape_left = shape
                    idx_u_left = i
                    max_shape_covar_left = covar
            except serums.errors.DistributionEstimatorFailed:
                pass

        if max_shape_left > 0:
            gamma_left = max_shape_left
            u_left = data_sorted[idx_u_left]
        else:
            raise serums.errors.OverboundingMethodFailed(
                "MLE indicates exponential or finite left tail. Use the paired Gaussian Overbounder."
            )

        Nt_left = idx_u_left - 1
        gamma_left = gamma_left + t.ppf(0.975, Nt_left) * np.sqrt(
            max_shape_covar_left[1, 1]
        )

        idx_90p = int(np.floor(0.9 * n))
        idxs_cand_u_right = np.arange(idx_90p, n - 250, 1)

        max_shape_right = 0
        idx_u_right = None
        max_shape_covar_right = None
        for i in idxs_cand_u_right:
            try:
                shape, scale, covar = de.grimshaw_MLE(
                    np.add(
                        np.abs(np.subtract(data_sorted[i:], data_sorted[i])),
                        1e-14,
                    )
                )
                if shape > max_shape_right:
                    max_shape_right = shape
                    idx_u_right = i
                    max_shape_covar_right = covar
            except serums.errors.DistributionEstimatorFailed:
                pass

        if max_shape_right > 0:
            gamma_right = max_shape_right
            u_right = data_sorted[idx_u_right]
        else:
            raise serums.errors.OverboundingMethodFailed(
                "MLE indicates exponential or finite right tail. Use the paired Gaussian Overbounder."
            )

        Nt_right = n - idx_u_right - 1
        gamma_right = gamma_right + t.ppf(0.975, Nt_right) * np.sqrt(
            max_shape_covar_right[1, 1]
        )

        # Define initial guess for basin hopping algorithm
        init_mean = np.mean(data)
        init_sigma = np.std(data, ddof=1)
        init_guess = np.array([init_mean, init_sigma])

        # Generate sample ECDF ordinates
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        # Compute Upper and Lower 95% DKW Confidence Bounds
        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        DKW_low = np.subtract(ecdf_ords, epsilon)
        DKW_high = np.add(ecdf_ords, epsilon)

        if self.StrictPairedEnforcement is True:
            left_usable_idxs = np.asarray(DKW_high < (1 - epsilon)).nonzero()[
                0
            ]
            left_usable_idxs = left_usable_idxs[idx_u_left:]
            x_check_left = data_sorted[left_usable_idxs]
            y_check_left = DKW_high[left_usable_idxs]
        else:
            left_usable_idxs = np.asarray(
                DKW_high < (0.5 + epsilon)
            ).nonzero()[0]
            left_usable_idxs = left_usable_idxs[idx_u_left:]
            x_check_left = data_sorted[left_usable_idxs]
            y_check_left = DKW_high[left_usable_idxs]

        left_result = basinhopping(
            self._cost_left,
            init_guess,
            niter=300,
            stepsize=np.array([init_mean / 2, init_sigma / 2]),
            minimizer_kwargs={
                "args": (x_check_left, y_check_left),
                "method": "Powell",
                "options": {
                    "xtol": 1e-14,
                    "ftol": 1e-6,
                    "maxfev": 10000,
                    "maxiter": 10000,
                },
            },
        )

        left_mean = left_result.x[0]
        left_sigma = left_result.x[1]

        if self.StrictPairedEnforcement is True:
            right_usable_idxs = np.asarray(DKW_low > epsilon).nonzero()[0]
            right_usable_idxs = right_usable_idxs[0 : -(n - 1 - idx_u_right)]
            x_check_right = data_sorted[right_usable_idxs]
            y_check_right = DKW_low[right_usable_idxs]
        else:
            right_usable_idxs = np.asarray(
                DKW_low > (0.5 - epsilon)
            ).nonzero()[0]
            right_usable_idxs = right_usable_idxs[0 : -(n - 1 - idx_u_right)]
            x_check_right = data_sorted[right_usable_idxs]
            y_check_right = DKW_low[right_usable_idxs]

        right_result = basinhopping(
            self._cost_right,
            init_guess,
            niter=300,
            stepsize=np.array([init_mean / 2, init_sigma / 2]),
            minimizer_kwargs={
                "args": (x_check_right, y_check_right),
                "method": "Powell",
                "options": {
                    "xtol": 1e-14,
                    "ftol": 1e-6,
                    "maxfev": 10000,
                    "maxiter": 10000,
                },
            },
        )

        right_mean = right_result.x[0]
        right_sigma = right_result.x[1]

        Fu = norm.cdf(u_left, loc=left_mean, scale=left_sigma)
        left_transformed_ords = np.divide(
            np.negative(
                np.subtract(
                    DKW_high[0 : idx_u_left - 1],
                    Fu,
                )
            ),
            Fu,
        )
        left_transformed_ords = np.flip(left_transformed_ords)
        shifted_left_tail = np.flip(
            np.abs(np.subtract(data_sorted[0 : idx_u_left - 1], u_left))
        )

        max_beta_left = 0

        for i in range(left_transformed_ords.size):
            beta = self.get_pareto_scale(
                shifted_left_tail[i], left_transformed_ords[i], gamma_left
            )
            if beta > max_beta_left:
                max_beta_left = beta

        if max_beta_left > 0:
            beta_left = max_beta_left
        else:
            raise (
                serums.errors.OverboundingMethodFailed(
                    "GPD scale parameter not found for left tail. Use the paired gaussian overbounder."
                )
            )

        Fu = norm.cdf(u_right, loc=right_mean, scale=right_sigma)
        right_transformed_ords = np.divide(
            np.abs(np.subtract(DKW_low[idx_u_right + 1 :], Fu)), (1 - Fu)
        )
        shifted_right_tail = np.abs(
            np.subtract(data_sorted[idx_u_right + 1 :], u_right)
        )

        max_beta_right = 0

        for i in range(right_transformed_ords.size):
            beta = self.get_pareto_scale(
                shifted_right_tail[i], right_transformed_ords[i], gamma_right
            )
            if beta > max_beta_right:
                max_beta_right = beta

        if max_beta_right > 0:
            beta_right = max_beta_right
        else:
            raise (
                serums.errors.OverboundingMethodFailed(
                    "GPD scale parameter not found for right tail. Use the paired gaussian overbounder"
                )
            )

        return smodels.PairedGaussianPareto(
            left_tail_shape=gamma_left,
            left_tail_scale=beta_left,
            left_threshold=u_left,
            left_mean=left_mean,
            left_sigma=left_sigma,
            right_tail_shape=gamma_right,
            right_tail_scale=beta_right,
            right_threshold=u_right,
            right_mean=right_mean,
            right_sigma=right_sigma,
        )
