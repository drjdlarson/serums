"""For SERUMS Overbound Estimation."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp
import serums.models as smodels
import serums.distribution_estimator as de
from abc import abstractmethod, ABC
from scipy.stats import norm, halfnorm, genpareto, t
import matplotlib.pyplot as plt
import math
import serums.errors
import time


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
            candidates[i] = self.find_sigma_h(abs_data[i], DKW_low[i])

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
        Nt_max = math.ceil(0.1 * n)
        idx_u_min = n - Nt_max - 1
        idx_u_max = n - Nt_min - 1
        u_idxs = np.arange(idx_u_min, idx_u_max + 1, 1)
        u_candidates = sorted_abs_data[u_idxs]
        Nu = u_candidates.size
        shapes = np.zeros(Nu)
        MLE_used = np.zeros(Nu, dtype=bool)

        print("\nComputing Tail GPD Shape Parameter Estimates...\nProgress:")

        resolution = 10 * (1 / Nu)
        for i in range(Nu):
            if i < Nu - 1:
                fraction = i / Nu
                checkpoint = math.floor(10 * fraction)
                diagnostic = (10 * fraction) - checkpoint
                if diagnostic < resolution:
                    print("{:3d}%".format(checkpoint * 10))
            else:
                print("100 %")

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
                sorted_abs_data[u_idxs[idx_shape_max[0]]],
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
        return (
            tail_shape,
            tail_scale,
            u,
            core_sigma,
        )
