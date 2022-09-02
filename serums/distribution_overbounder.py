"""For SERUMS Overbound Estimation."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp
import serums.models
from abc import abstractmethod, ABC
from scipy.stats import norm
import matplotlib.pyplot as plt


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

    @abstractmethod
    def overbound(self, *args, **kwargs):
        """Produce the overbound of the type denoted by the child class name."""
        raise NotImplementedError("not implemented")


class SymmetricGaussian(OverbounderBase):
    """Represents a Symmetric Gaussian Overbound object."""

    def __init__(self):
        super().__init__()
        self.pierce_locator = 1

    def overbound(self, data):
        """Produce Symmetric Gaussian Overbound of empirical error data."""
        n = data.size
        locator = 1
        sample_sigma = np.std(data, ddof=1)
        threshold = locator * sample_sigma
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

        return OB_mean, OB_sigma


test_instance = SymmetricGaussian()
