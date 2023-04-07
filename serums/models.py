"""Defines various distribution models."""
from __future__ import annotations
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
from warnings import warn
import matplotlib.pyplot as plt
from scipy.stats import norm, halfnorm, genpareto
from copy import deepcopy
import probscale

import serums.enums as enums


class BaseSingleModel:
    """Generic base class for distribution models.

    This defines the required functions and provides their recommended function
    signature for inherited classes. It also defines base attributes for the
    distribution.

    Attributes
    ----------
    location : N x 1 numpy array
        location parameter of the distribution
    scale : N x N numpy array
        scale parameter of the distribution
    """

    def __init__(self, loc=None, scale=None, monte_carlo_size=int(1e4)):
        super().__init__()
        if isinstance(loc, np.ndarray):
            self.location = loc
        else:
            self.location = np.array([[loc]])

        if isinstance(scale, np.ndarray):
            self.scale = scale
        else:
            self.scale = np.array([[scale]])

        self.monte_carlo_size = monte_carlo_size

    def sample(self, rng: rnd._generator = None, num_samples: int = None) -> np.ndarray:
        """Draw a sample from the distribution.

        This should be implemented by the child class.

        Parameters
        ----------
        rng : numpy random generator, optional
            random number generator to use. The default is None.

        Returns
        -------
        None.
        """
        warn("sample not implemented by class {}".format(type(self).__name__))

    def pdf(self, x):
        """Calculate the PDF value at the given point.

        This should be implemented by the child class.

        Parameters
        ----------
        x : N x 1 numpy array
            Point to evaluate the PDF.

        Returns
        -------
        float
            PDF value.
        """
        warn("pdf not implemented by class {}".format(type(self).__name__))
        return np.nan

    def __str__(self):
        msg = "Location = "
        dim = self.mean.size
        for ii in range(dim):
            if ii != 0:
                msg += "{:11s}".format("")

            if ii == 0 and dim != 1:
                fmt = "\u2308{:.4e}\u2309\tScale = \u2308"
                fmt += "{:.4e}, " * (dim - 1) + "{:.4e}" + "\u2309"
            elif ii == (dim - 1) and dim != 1:
                fmt = (
                    "\u230A{:.4e}\u230B\t"
                    + "{:8s}\u230A".format("")
                    + "{:.4e}, " * (dim - 1)
                    + "{:.4e}"
                    + "\u230B"
                )
            else:
                fmt = "|{:.4e}|\t"
                if dim == 1:
                    fmt += "Scale = |"
                else:
                    fmt += "{:8s}|".format("")
                fmt += "{:.4e}, " * (dim - 1) + "{:.4e}" + "|"
            msg += (
                fmt.format(self.mean.ravel()[ii], *self.covariance[ii, :].tolist())
                + "\n"
            )
        return msg

    def __sub__(self, other: BaseSingleModel | float | int) -> np.ndarray:
        if isinstance(other, BaseSingleModel):
            if self.location.size != other.location.size:
                raise RuntimeError(
                    "Can not subtract distributions of different shapes ({:d} vs {:d})".format(
                        self.location.size, other.location.size
                    )
                )
            n_samps = np.max([self.monte_carlo_size, other.monte_carlo_size]).astype(
                int
            )
            return self.sample(num_samples=n_samps) - other.sample(num_samples=n_samps)
        else:
            return self.sample(num_samples=self.monte_carlo_size) - other

    def __neg__(self) -> np.ndarray:
        """Should only be used to redefine order of operations for a subtraction operation (i.e. -g1 + g2 vs g2 - g1)"""
        return -self.sample(num_samples=int(self.monte_carlo_size))

    def __add__(self, other: BaseSingleModel | float | int) -> np.ndarray:
        if isinstance(other, BaseSingleModel):
            if self.location.size != other.location.size:
                raise RuntimeError(
                    "Can not add distributions of different shapes ({:d} vs {:d})".format(
                        self.location.size, other.location.size
                    )
                )
            n_samps = np.max([self.monte_carlo_size, other.monte_carlo_size]).astype(
                int
            )
            return self.sample(num_samples=n_samps) + other.sample(num_samples=n_samps)
        else:
            return self.sample(num_samples=self.monte_carlo_size) + other

    # define right multiplication to be the same as normal multiplication (allow scalar * distribution)
    def __rmul__(self, other: BaseSingleModel | float | int) -> np.ndarray:
        return self.__mul__(other)

    def __mul__(self, other: BaseSingleModel | float | int) -> np.ndarray:
        if isinstance(other, BaseSingleModel):
            if self.location.size != other.location.size:
                raise RuntimeError(
                    "Can not multiply distributions of different shapes ({:d} vs {:d})".format(
                        self.location.size, other.location.size
                    )
                )
            n_samps = np.max([self.monte_carlo_size, other.monte_carlo_size]).astype(
                int
            )
            return self.sample(num_samples=n_samps) * other.sample(num_samples=n_samps)
        else:
            return self.sample(num_samples=self.monte_carlo_size) * other

    def __truediv__(self, other: BaseSingleModel | float | int) -> np.ndarray:
        if isinstance(other, BaseSingleModel):
            if self.location.size != other.location.size:
                raise RuntimeError(
                    "Can not divide distributions of different shapes ({:d} vs {:d})".format(
                        self.location.size, other.location.size
                    )
                )
            n_samps = np.max([self.monte_carlo_size, other.monte_carlo_size]).astype(
                int
            )
            return self.sample(num_samples=n_samps) / other.sample(num_samples=n_samps)
        else:
            return self.sample(num_samples=self.monte_carlo_size) / other

    def __rtruediv__(self, other: float | int) -> np.ndarray:
        return other / self.sample(num_samples=self.monte_carlo_size)

    def __floordiv__(self, other: BaseSingleModel | float | int) -> np.ndarray:
        if isinstance(other, BaseSingleModel):
            if self.location.size != other.location.size:
                raise RuntimeError(
                    "Can not floor divide distributions of different shapes ({:d} vs {:d})".format(
                        self.location.size, other.location.size
                    )
                )
            n_samps = np.max([self.monte_carlo_size, other.monte_carlo_size]).astype(
                int
            )
            return self.sample(num_samples=n_samps) // other.sample(num_samples=n_samps)
        else:
            return self.sample(num_samples=self.monte_carlo_size) // other

    def __rfloordiv__(self, other: float | int) -> np.ndarray:
        return other // self.sample(num_samples=self.monte_carlo_size)

    def __pow__(self, power: int) -> np.ndarray:
        return self.sample(num_samples=int(self.monte_carlo_size)) ** power


class Gaussian(BaseSingleModel):
    """Represents a Gaussian distribution object."""

    def __init__(self, mean=None, covariance=None):
        """Initialize an object.

        Parameters
        ----------
        mean : N x 1 numpy array, optional
            Mean of the distribution. The default is None.
        covariance : N x N numpy array, optional
            Covariance of the distribution. The default is None.

        Returns
        -------
        None.
        """
        super().__init__(loc=mean, scale=covariance)
        if covariance is not None:
            try:
                self.stdev = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError:
                self.stdev = None
        else:
            self.stdev = None

    def __str__(self):
        msg = "Mean = "
        dim = self.mean.size
        for ii in range(dim):
            if ii != 0:
                msg += "{:7s}".format("")

            if ii == 0 and dim != 1:
                fmt = "\u2308{:.4e}\u2309\tCovariance = \u2308"
                fmt += "{:.4e}, " * (dim - 1) + "{:.4e}" + "\u2309"
            elif ii == (dim - 1) and dim != 1:
                fmt = (
                    "\u230A{:.4e}\u230B\t"
                    + "{:13s}\u230A".format("")
                    + "{:.4e}, " * (dim - 1)
                    + "{:.4e}"
                    + "\u230B"
                )
            else:
                fmt = "|{:.4e}|\t"
                if dim == 1:
                    fmt += "Covariance = |"
                else:
                    fmt += "{:13s}|".format("")
                fmt += "{:.4e}, " * (dim - 1) + "{:.4e}" + "|"
            msg += (
                fmt.format(self.mean.ravel()[ii], *self.covariance[ii, :].tolist())
                + "\n"
            )
        return msg

    @property
    def mean(self):
        """Mean of the distribution.

        Returns
        -------
        N x 1 numpy array.
        """
        return self.location

    @mean.setter
    def mean(self, val):
        self.location = val

    @property
    def covariance(self):
        """Covariance of the distribution.

        Returns
        -------
        N x N numpy array.
        """
        return self.scale

    @covariance.setter
    def covariance(self, val):
        self.scale = val
        try:
            self.stdev = np.linalg.cholesky(val)
        except np.linalg.LinAlgError:
            self.stdev = None

    def sample(self, rng=None, num_samples=None):
        """Draw a sample from the current mixture model.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        numpy array
            randomly sampled numpy array of the same shape as the mean.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1
        return rng.multivariate_normal(
            self.mean.flatten(), self.covariance, size=int(num_samples)
        )

    def pdf(self, x):
        """Multi-variate probability density function for this distribution.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.multivariate_normal
        return rv.pdf(x.flatten(), mean=self.mean.flatten(), cov=self.covariance)

    def CI(self, alfa):
        """Return confidence interval of distribution given a significance level 'alfa'.

        Parameters
        ----------
        alfa : float
            significance level, i.e. confidence level = (1 - alfa). Must be
            a positive real number which is less than 1

        Returns
        -------
        1 x 2 numpy array
            Numpy array containing the upper and lower bound of the computed
            confidence interval.
        """
        p = alfa / 2
        low = stats.norm.ppf(p, loc=self.mean, scale=np.sqrt(self.scale))
        high = stats.norm.ppf(1 - p, loc=self.mean, scale=np.sqrt(self.scale))
        return np.array([[low[0, 0], high[0, 0]]])

    def CDFplot(self, data):
        """Plot the overbound and DKW bound(s) against ECDF of input data.

        Parameters
        ----------
        data : N numpy array
            Contains the error sample data for which the overbound was computed.

        Returns
        -------
        matplotlib line plot
            Shows empirical cumulative distribution function of input error
            data, the associated DKW bound(s), and the computed overbound in the
            CDF domain.
        """
        n = data.size
        ordered_abs_data = np.sort(np.abs(data))
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        X_ECDF = ordered_abs_data
        X_OB = ordered_abs_data
        Y_ECDF = ecdf_ords
        Y_OB = halfnorm.cdf(X_OB, loc=0, scale=np.sqrt(self.scale[0, 0]))

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        plt.figure("Symmetric Gaussian Overbound Plot in Half-Gaussian CDF Domain")
        plt.plot(X_ECDF, Y_ECDF, label="Original ECDF")
        plt.plot(X_ECDF, np.add(Y_ECDF, epsilon), label="DKW Upper Band")
        plt.plot(X_ECDF, np.subtract(Y_ECDF, epsilon), label="DKW Lower Band")
        plt.plot(X_OB, Y_OB, label="Overbound CDF")
        plt.xlim(np.array([0, 1.1 * ordered_abs_data[-1]]))
        plt.ylim(np.array([0, 1]))
        plt.legend()
        plt.grid()
        plt.title("Symmetric Gaussian Overbound Plot in CDF Domain")
        plt.ylabel("Accumulated Probability")
        plt.xlabel("Error Magnitude")

    def probscaleplot(self, data):
        """Generate probability plot of the ECDF, overbound, and DKW bound(s).

        Parameters
        ----------
        data : N numpy array
            numpy array containing the error data used to calculate the
            overbound

        Returns
        -------
        matplotlib line plot
            Shows empirical cumulative distribution function of input error
            data, the associated DKW bound(s), and the computed overbound
            in the CDF domain where the probability axis is represented with
            percentiles and is scaled such that a Gaussian CDF is linear.
        """
        sorted_abs_data = np.sort(np.abs(data))
        n = data.size
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n
        x_ecdf = sorted_abs_data
        y_ecdf = ecdf_ords

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))

        x_dkw = x_ecdf
        y_dkw = np.subtract(y_ecdf, epsilon)

        x_ob = sorted_abs_data
        y_ob = halfnorm.cdf(x_ob, loc=0, scale=np.sqrt(self.scale[0, 0]))
        dist_type = halfnorm()

        figure, ax = plt.subplots()
        ax.set_ylim(bottom=5, top=99.999)
        ax.set_yscale("prob", dist=dist_type)

        plt.plot(x_ecdf, 100 * y_ecdf)
        plt.plot(x_dkw, 100 * y_dkw)
        plt.plot(x_ob, 100 * y_ob)
        plt.legend(["ECDF", "DKW Lower Bound", "Symmetric Gaussian Overbound"])
        plt.title("Probability Plot of Symmetric Gaussian Overbound")
        plt.ylabel("CDF Percentiles")
        plt.xlabel("Error Magnitude")
        plt.grid()


class PairedGaussian(BaseSingleModel):
    """Represents a Paired Gaussian Overbound Distribution Object."""

    def __init__(self, left: Gaussian, right: Gaussian):
        """Initialize an object.

        Parameters
        ----------
        left : :class:'serums.models.Gaussian'
            Gaussian model storing the parameters of the left component of the
            paired overbound.
        right : :class:'serums.models.Gaussian'
            Gaussian model storing the parameters of the right component of the
            paired overbound.

        Returns
        -------
        None.
        """
        super().__init__()

        self.left_gaussian = deepcopy(left)
        self.right_gaussian = deepcopy(right)

    def sample(self, rng: rnd._generator = None, num_samples: int = None) -> np.ndarray:
        """Generate a random sample from the distribution model.

        Parameters
        ----------
        num_samples : int
            Specify the size of the sample.

        Returns
        -------
        N numpy array
            Numpy array containing a random sample of the specified size from
            the distribution.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1

        p = rng.uniform(size=num_samples)
        rcount = int(np.sum(p[p >= 0.5]))
        lcount = int(num_samples) - rcount

        samp = np.nan * np.ones((num_samples, self.right_gaussian.mean.size))
        if rcount > 0:
            rsamp = self.right_gaussian.sample(rng=rng, num_samples=int(rcount))
            samp[0:rcount] = (
                np.abs(rsamp - self.right_gaussian.mean) + self.right_gaussian.mean
            )

        if lcount > 0:
            lsamp = self.left_gaussian.sample(rng=rng, num_samples=int(lcount))
            samp[rcount:] = (
                -np.abs(lsamp - self.left_gaussian.mean) + self.left_gaussian.mean
            )

        return samp

    def CI(self, alfa):
        """Return confidence interval of distribution given a significance level 'alfa'.

        Parameters
        ----------
        alfa : float
            significance level, i.e. confidence level = (1 - alfa). Must be
            a positive real number which is less than 1

        Returns
        -------
        1 x 2 numpy array
            Numpy array containing the upper and lower bound of the computed
            confidence interval.
        """
        e = alfa / 2
        left = norm.ppf(
            e,
            loc=self.left_gaussian.location,
            scale=np.sqrt(self.left_gaussian.scale),
        )
        right = norm.ppf(
            1 - e,
            loc=self.right_gaussian.location,
            scale=np.sqrt(self.right_gaussian.scale),
        )
        return np.array([[left, right]])

    def CDFplot(self, data):
        """Plot the overbound and DKW bound(s) against ECDF of input data.

        Parameters
        ----------
        data : N numpy array
            Contains the error sample data for which the overbound was computed.

        Returns
        -------
        matplotlib line plot
            Shows empirical cumulative distribution function of input error
            data, the associated DKW bound(s), and the computed overbound in the
            CDF domain.
        """
        data = np.sort(data)
        n = data.size
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        DKW_low = np.subtract(ecdf_ords, epsilon)
        DKW_high = np.add(ecdf_ords, epsilon)

        left_mean = self.left_gaussian.mean
        left_std = np.sqrt(self.left_gaussian.covariance)
        right_mean = self.right_gaussian.mean
        right_std = np.sqrt(self.right_gaussian.covariance)

        y_left_ob = np.reshape(norm.cdf(data, loc=left_mean, scale=left_std), (n,))
        y_right_ob = np.reshape(norm.cdf(data, loc=right_mean, scale=right_std), (n,))
        x_paired_ob = np.linspace(np.min(data) - 1, np.max(data) + 1, num=10000)
        y_paired_ob = np.zeros(x_paired_ob.size)
        left_pt = self.left_gaussian.mean
        right_pt = self.right_gaussian.mean

        for i in range(y_paired_ob.size):
            if x_paired_ob[i] < left_pt:
                y_paired_ob[i] = norm.cdf(x_paired_ob[i], loc=left_mean, scale=left_std)
            elif x_paired_ob[i] > right_pt:
                y_paired_ob[i] = norm.cdf(
                    x_paired_ob[i], loc=right_mean, scale=right_std
                )
            else:
                y_paired_ob[i] = 0.5

        plt.figure("Paired Overbound in CDF Domain")
        plt.plot(data, y_left_ob, label="Left OB", linestyle="--")
        plt.plot(data, y_right_ob, label="Right OB", linestyle="--")
        plt.plot(x_paired_ob, y_paired_ob, label="Paired OB")
        plt.plot(data, ecdf_ords, label="ECDF")
        plt.plot(data, DKW_high, label="Upper DKW Bound")
        plt.plot(data, DKW_low, label="Lower DKW Bound")
        plt.legend()
        plt.grid()
        plt.title("Paired Gaussian Overbound Plot in CDF Domain")
        plt.ylabel("Accumulated Probability")
        plt.xlabel("Error")

    def probscaleplot(self, data):
        """Generate probability plot of the ECDF, overbound, and DKW bound(s).

        Parameters
        ----------
        data : N numpy array
            numpy array containing the error data used to calculate the
            overbound

        Returns
        -------
        matplotlib line plot
            Shows empirical cumulative distribution function of input error
            data, the associated DKW bound(s), and the computed overbound
            in the CDF domain where the probability axis is represented with
            percentiles and is scaled such that a Gaussian CDF is linear.
        """
        n = data.size
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        DKW_low = np.subtract(ecdf_ords, epsilon)
        DKW_high = np.add(ecdf_ords, epsilon)

        left_mean = self.left_gaussian.mean
        left_std = np.sqrt(self.left_gaussian.covariance)
        right_mean = self.right_gaussian.mean
        right_std = np.sqrt(self.right_gaussian.covariance)

        y_left_ob = np.reshape(norm.cdf(data, loc=left_mean, scale=left_std), (n,))
        y_right_ob = np.reshape(norm.cdf(data, loc=right_mean, scale=right_std), (n,))
        x_paired_ob = np.linspace(np.min(data), np.max(data), num=10000)
        y_paired_ob = np.zeros(x_paired_ob.size)
        left_pt = self.left_gaussian.mean
        right_pt = self.right_gaussian.mean

        for i in range(y_paired_ob.size):
            if x_paired_ob[i] < left_pt:
                y_paired_ob[i] = norm.cdf(x_paired_ob[i], loc=left_mean, scale=left_std)
            elif x_paired_ob[i] > right_pt:
                y_paired_ob[i] = norm.cdf(
                    x_paired_ob[i], loc=right_mean, scale=right_std
                )
            else:
                y_paired_ob[i] = 0.5

        dist_type = norm()

        x_ecdf = np.sort(data)
        y_ecdf = ecdf_ords

        x_dkw_low = x_ecdf
        y_dkw_low = DKW_low

        x_dkw_high = x_ecdf
        y_dkw_high = DKW_high

        figure, ax = plt.subplots()
        ax.set_ylim(bottom=0.001, top=99.999)
        ax.set_yscale("prob", dist=dist_type)

        plt.plot(x_ecdf, 100 * y_ecdf)
        plt.plot(x_dkw_low, 100 * y_dkw_low)
        plt.plot(x_dkw_high, 100 * y_dkw_high)
        plt.plot(x_paired_ob, 100 * y_paired_ob)
        plt.legend(
            [
                "ECDF",
                "DKW Lower Bound",
                "DKW Upper Bound",
                "Symmetric Gaussian Overbound",
            ]
        )
        plt.title("Probability Plot of Paired Gaussian Overbound")
        plt.ylabel("CDF Percentiles")
        plt.xlabel("Error")
        plt.grid()


class StudentsT(BaseSingleModel):
    """Represents a Student's t-distribution."""

    def __init__(self, mean=None, scale=None, dof=None):
        super().__init__(loc=mean, scale=scale)
        self._dof = dof

    @property
    def mean(self):
        """Mean of the distribution.

        Returns
        -------
        N x 1 numpy array.
        """
        return self.location

    @mean.setter
    def mean(self, val):
        self.location = val

    @property
    def degrees_of_freedom(self):
        """Degrees of freedom of the distribution, must be greater than 0."""
        return self._dof

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, value):
        self._dof = value

    @property
    def covariance(self):
        """Read only covariance of the distribution (if defined).

        Returns
        -------
        N x N numpy array.
        """
        if self._dof <= 2:
            msg = "Degrees of freedom is {} and must be > 2"
            raise RuntimeError(msg.format(self._dof))
        return self._dof / (self._dof - 2) * self.scale

    @covariance.setter
    def covariance(self, val):
        warn("Covariance is read only.")

    def pdf(self, x):
        """Multi-variate probability density function for this distribution.

        Parameters
        ----------
        x : N x 1 numpy array
            Value to evaluate the pdf at.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.multivariate_t
        return rv.pdf(
            x.flatten(),
            loc=self.location.flatten(),
            shape=self.scale,
            df=self.degrees_of_freedom,
        )

    def sample(self, rng=None, num_samples=None):
        """Multi-variate probability density function for this distribution.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1

        rv = stats.multivariate_t
        rv.random_state = rng
        x = rv.rvs(
            loc=self.location.flatten(),
            shape=self.scale,
            df=self.degrees_of_freedom,
            size=int(num_samples),
        )

        return x.reshape((x.size, 1))


class ChiSquared(BaseSingleModel):
    """Represents a Chi Squared distribution."""

    def __init__(self, mean=None, scale=None, dof=None):
        super().__init__(loc=mean, scale=scale)
        self._dof = dof

    @property
    def mean(self):
        """Mean of the distribution.

        Returns
        -------
        N x 1 numpy array.
        """
        return self.location

    @mean.setter
    def mean(self, val):
        self.location = val

    @property
    def degrees_of_freedom(self):
        """Degrees of freedom of the distribution, must be greater than 0."""
        return self._dof

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, value):
        self._dof = value

    @property
    def covariance(self):
        """Read only covariance of the distribution (if defined).

        Returns
        -------
        N x N numpy array.
        """
        if self._dof < 0:
            msg = "Degrees of freedom is {} and must be > 0"
            raise RuntimeError(msg.format(self._dof))
        return (self._dof * 2) * (self.scale**2)

    @covariance.setter
    def covariance(self, val):
        warn("Covariance is read only.")

    def pdf(self, x):
        """Multi-variate probability density function for this distribution.

        Parameters
        ----------
        x : N x 1 numpy array
            Value to evaluate the pdf at.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.chi2
        return rv.pdf(
            x.flatten(),
            self._dof,
            loc=self.location.flatten(),
            shape=self.scale,
        )

    def sample(self, rng=None, num_samples=None):
        """Multi-variate probability density function for this distribution.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1

        rv = stats.chi2
        rv.random_state = rng
        x = rv.rvs(
            self._dof,
            loc=self.location.flatten(),
            scale=self.scale,
            size=int(num_samples),
        )
        if num_samples == 1:
            return x.reshape((-1, 1))
        else:
            return x


class Cauchy(StudentsT):
    """Represents a Cauchy distribution.

    This is a special case of the Student's t-distribution with the degrees of
    freedom fixed at 1. However, the mean and covariance do not exist for this
    distribution.
    """

    def __init__(self, location=None, scale=None):
        super().__init__(scale=scale, dof=1)
        self.location = location

    @property
    def mean(self):
        """Mean of the distribution."""
        warn("Mean does not exist for a Cauchy")

    @mean.setter
    def mean(self, val):
        warn("Mean does not exist for a Cauchy")

    @property
    def degrees_of_freedom(self):
        """Degrees of freedom of the distribution, fixed at 1."""
        return super().degrees_of_freedom

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, value):
        warn("Degrees of freedom is 1 for a Cauchy")

    @property
    def covariance(self):
        """Read only covariance of the distribution (if defined)."""
        warn("Covariance is does not exist.")

    @covariance.setter
    def covariance(self, val):
        warn("Covariance is does not exist.")


class GaussianScaleMixture(BaseSingleModel):
    r"""Helper class for defining Gaussian Scale Mixture objects.

    Note
    ----
    This is an alternative method for representing heavy-tailed distributions
    by modeling them as a combination of a standard Gaussian, :math:`v`, and
    another positive random variable known as the generating variate, :math:`z`

    .. math::
        x \overset{d}{=} \sqrt{z} v

    where :math:`\overset{d}{=}` means equal in distribution and :math:`x`
    follows a GSM distribution (in general, a heavy tailed distribution).
    This formulation is based on
    :cite:`VilaValls2012_NonlinearBayesianFilteringintheGaussianScaleMixtureContext`,
    :cite:`Wainwright1999_ScaleMixturesofGaussiansandtheStatisticsofNaturalImages`, and
    :cite:`Kuruoglu1998_ApproximationofAStableProbabilityDensitiesUsingFiniteGaussianMixtures`.

    Attributes
    ----------
    type : :class:`serums.enums.GSMTypes`
        Type of the distribution to represent as a GSM.
    location_range : tuple
        Minimum and maximum values for the location parameter. Useful if being
        fed to a filter for estimating the location parameter. Each element must
        match the type of the :attr:`BaseSingleModel.location` attribute.
    scale_range : tuple
        Minimum and maximum values for the scale parameter. Useful if being
        fed to a filter for estimating the scale parameter. Each element must
        match the type of the :attr:`BaseSingleModel.scale` attribute. The default is None.
    df_range : tuple
        Minimum and maximum values for the degree of freedom parameter.
        Useful if being fed to a filter for estimating the degree of freedom
        parameter. Each element must be a float. The default is None.
    """

    __df_types = (enums.GSMTypes.STUDENTS_T, enums.GSMTypes.CAUCHY)

    def __init__(
        self,
        gsm_type,
        location=None,
        location_range=None,
        scale=None,
        scale_range=None,
        degrees_of_freedom=None,
        df_range=None,
    ):
        """Initialize a GSM Object.

        Parameters
        ----------
        gsm_type : :class:`serums.enums.GSMTypes`
            Type of the distribution to represent as a GSM.
        location : N x 1 numpy array, optional
            location parameter of the distribution. The default is None.
        location_range : tuple, optional
            Minimum and maximum values for the location parameter. Useful if being
            fed to a filter for estimating the location parameter. Each element must
            match the type of the :attr:`BaseSingleModel.location` attribute. The default is None
        scale : N x N numpy array, optional
            Scale parameter of the distribution being represented as a GSM.
            The default is None.
        scale_range : tuple, optional
            Minimum and maximum values for the scale parameter. Useful if being
            fed to a filter for estimating the scale parameter. Each element must
            match the type of the :attr:`BaseSingleModel.scale` attribute. The default is None.
        degrees_of_freedom : float, optional
            Degrees of freedom parameter of the distribution being represented
            as a GSM. This is not needed by all types. The default is None.
        df_range : tuple, optional
            Minimum and maximum values for the degree of freedom parameter.
            Useful if being fed to a filter for estimating the degree of freedom
            parameter. Each element must be a float. The default is None.

        Raises
        ------
        RuntimeError
            If a `gsm_type` is given that is of the incorrect data type.
        """
        super().__init__(loc=location, scale=scale)

        if not isinstance(gsm_type, enums.GSMTypes):
            raise RuntimeError("Type ({}) must be a GSMType".format(gsm_type))

        self.type = gsm_type

        self._df = None

        self.location_range = location_range
        self.scale_range = scale_range
        self.df_range = df_range

        if degrees_of_freedom is not None:
            self.degrees_of_freedom = degrees_of_freedom

        if self.type is enums.GSMTypes.CAUCHY:
            self._df = 1

    @property
    def degrees_of_freedom(self):
        """Degrees of freedom parameter of the distribution being represented as a GSM.

        Returns
        -------
        float, optional
        """
        if self.type in self.__df_types:
            return self._df
        else:
            msg = "GSM type {:s} does not have a degree of freedom.".format(self.type)
            warn(msg)
            return None

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, val):
        if self.type in self.__df_types:
            if self.type is enums.GSMTypes.CAUCHY:
                warn("GSM type {:s} requires degree of freedom = 1".format(self.type))
                return
            self._df = val
        else:
            msg = (
                "GSM type {:s} does not have a degree of freedom. " + "Skipping"
            ).format(self.type)
            warn(msg)

    def sample(self, rng=None, num_samples=None):
        """Draw a sample from the specified GSM type.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        float
            randomly sampled value from the GSM.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1

        if self.type in [enums.GSMTypes.STUDENTS_T, enums.GSMTypes.CAUCHY]:
            return self._sample_student_t(rng, int(num_samples))

        elif self.type is enums.GSMTypes.SYMMETRIC_A_STABLE:
            return self._sample_SaS(rng, int(num_samples))

        else:
            raise RuntimeError("GSM type: {} is not supported".format(self.type))

    def _sample_student_t(self, rng, num_samples=None):
        return stats.t.rvs(
            self.degrees_of_freedom,
            scale=self.scale,
            random_state=rng,
            size=num_samples,
        )

    def _sample_SaS(self, rng, num_samples):
        raise RuntimeError("sampling SaS distribution not implemented")


class GeneralizedPareto(BaseSingleModel):
    """Represents a Generalized Pareto distribution (GPD)."""

    def __init__(self, location=None, scale=None, shape=None):
        """Initialize an object.

        Parameters
        ----------
        location : N x 1 numpy array, optional
            location of the distribution. The default is None.
        scale : N x 1 numpy array, optional
            scale of the distribution. The default is None.
        shape : N x 1 numpy array, optional
            shape of the distribution. The default is None.

        Returns
        -------
        None.
        """
        super().__init__(loc=location, scale=scale)
        self.shape = shape

    def sample(self, rng=None, num_samples=None):
        """Draw a sample from the current mixture model.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        numpy array
            randomly sampled numpy array of the same shape as the mean.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1

        rv = stats.genpareto
        rv.random_state = rng
        x = (
            self.scale * rv.rvs(self.shape, size=int(num_samples))
        ) + self.location.ravel()
        if num_samples == 1:
            return x.reshape((-1, 1))
        else:
            return x

    def pdf(self, x):
        """Multi-variate probability density function for this distribution.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.genpareto
        return (
            rv.pdf(
                (x.flatten() - self.location) / self.scale,
                shape=self.shape.flatten(),
            )
            / self.scale
        )


class BaseMixtureModelIterator:
    """Iterator for :class:`serums.models.BaseMixutreModel`.

    Attributes
    ----------
    weights : list
        Each element is a float for the weight of a distribution.
    dists : list
        Each element is a :class:`serums.models.BaseSingleModel`.
    idx : int
        Current index for the iterator.
    """

    def __init__(self, weights, dists):
        """Initialize an object.

        Parameters
        ----------
        weights : list
            Each element is a float for the weight of a distribution.
        dists : list
            Each element is a :class:`serums.models.BaseSingleModel`.
        """
        self.weights = weights
        self.dists = dists
        self.idx = 0

    def __iter__(self):
        """Returns the iterator object."""
        return self

    def __next__(self):
        """Get the next element in the iterator.

        Raises
        ------
        StopIteration
            End of the iterator is reached.

        Returns
        -------
        float
            weight of the distribution.
        :class:`serums.models.BaseSingleModel`
            distribution object.
        """
        self.idx += 1
        try:
            return self.weights[self.idx - 1], self.dists[self.idx - 1]
        except IndexError:
            self.idx = 0
            raise StopIteration  # Done iterating.


class BaseMixtureModel:
    """Generic base class for mixture distribution models.

    This defines the required functions and provides their recommended function
    signature for inherited classes. It also defines base attributes for the
    mixture model.

    Attributes
    ----------
    weights : list
        weight of each distribution
    """

    def __init__(self, distributions=None, weights=None, monte_carlo_size=int(1e4)):
        """Initialize a mixture model object.

        Parameters
        ----------
        distributions : list, optional
            Each element is a :class:`.BaseSingleModel`. The default is None.
        weights : list, optional
            Weight of each distribution. The default is None.

        Returns
        -------
        None.

        """
        if distributions is None:
            distributions = []
        if weights is None:
            weights = []

        self._distributions = distributions
        self.weights = weights
        self.monte_carlo_size = monte_carlo_size

    def __iter__(self):
        """Allow iterating over mixture objects by (weight, distribution)."""
        return BaseMixtureModelIterator(self.weights, self._distributions)

    def __len__(self):
        """Give the number of distributions in the mixture."""
        return len(self._distributions)

    def sample(self, rng=None, num_samples=None):
        """Draw a sample from the current mixture model.

        Parameters
        ----------
        rng : numpy random generator, optional
            Random number generator to use. If none is given then the numpy
            default is used. The default is None.

        Returns
        -------
        numpy array
            randomly sampled numpy array of the same shape as the mean.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1
        if num_samples == 1:
            mix_ind = rng.choice(np.arange(len(self), dtype=int), p=self.weights)
            x = self._distributions[mix_ind].sample(rng=rng)
            return x.reshape((x.size, 1))
        else:
            x = np.nan * np.ones(
                (int(num_samples), self._distributions[0].location.size())
            )
            for ii in range(int(num_samples)):
                mix_ind = rng.choice(np.arange(len(self), dtype=int), p=self.weights)
                x[ii, :] = self._distributions[mix_ind].sample(rng=rng).ravel()
            return x

    def pdf(self, x):
        """Multi-variate probability density function for this mixture.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        p = 0
        for w, dist in self:
            p += w * dist.pdf(x)

        return p

    def remove_components(self, indices):
        """Remove component distributions from the mixture by index.

        Parameters
        ----------
        indices : list
            indices of distributions to remove.

        Returns
        -------
        None.
        """
        if not isinstance(indices, list):
            indices = list(indices)

        for index in sorted(indices, reverse=True):
            del self._distributions[index]
            del self.weights[index]

    def add_components(self, *args):
        """Add a component distribution to the mixture.

        This should be implemented by the child class.

        Parameters
        ----------
        *args : tuple
            Additional arguments specific to the child distribution.

        Returns
        -------
        None.
        """
        warn("add_component not implemented by {}".format(type(self).__name__))


class _DistListWrapper(list):
    """Helper class for wrapping lists of BaseSingleModel to get a list of a single parameter."""

    def __init__(self, dist_lst, attr):
        """Give list of distributions and the attribute to access."""
        self.dist_lst = dist_lst
        self.attr = attr

    def __getitem__(self, index):
        """Get the attribute of the item at the index in the list."""
        if isinstance(index, slice):
            step = 1
            if index.step is not None:
                step = index.step
            return [
                getattr(self.dist_lst[ii], self.attr)
                for ii in range(index.start, index.stop, step)
            ]
        elif isinstance(index, int):
            return getattr(self.dist_lst[index], self.attr)

        else:
            fmt = "Index must be a integer or slice not {}"
            raise RuntimeError(fmt.format(type(index)))

    def __setitem__(self, index, val):
        """Set the attribute of the item at the index to the value."""
        if isinstance(index, slice):
            step = 1
            if index.step is not None:
                step = index.step
            for ii in range(index.start, index.stop, step):
                setattr(self.dist_lst[ii], self.attr, val)

        elif isinstance(index, int):
            setattr(self.dist_lst[index], self.attr, val)

        else:
            fmt = "Index must be a integer or slice not {}"
            raise RuntimeError(fmt.format(type(index)))

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.dist_lst):
            self.n += 1
            return getattr(self.dist_lst[self.n - 1], self.attr)
        else:
            raise StopIteration

    def __repr__(self):
        return str([getattr(d, self.attr) for d in self.dist_lst])

    def __len__(self):
        return len(self.dist_lst)

    def append(self, *args):
        raise RuntimeError("Cannot append, use add_component function instead.")

    def extend(self, *args):
        raise RuntimeError("Cannot extend, use add_component function instead.")


class GaussianMixture(BaseMixtureModel):
    """Gaussian Mixture object."""

    def __init__(self, means=None, covariances=None, **kwargs):
        """Initialize an object.

        Parameters
        ----------
        means : list, optional
            Each element is a N x 1 numpy array. Will be used in place of supplied
            distributions but requires covariances to also be given. The default is None.
        covariances : list, optional
            Each element is an N x N numpy array. Will be used in place of
            supplied distributions but requires means to be given. The default is None.
        **kwargs : dict, optional
            See the base class for details.

        Returns
        -------
        None.
        """
        if means is not None and covariances is not None:
            kwargs["distributions"] = [
                Gaussian(mean=m, covariance=c) for m, c in zip(means, covariances)
            ]
        super().__init__(**kwargs)

    @property
    def means(self):
        """List of Gaussian means, each is a N x 1 numpy array. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "location")

    @means.setter
    def means(self, val):
        if not isinstance(val, list):
            warn("Must set means to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for ii in range(len(val))]
            self._distributions = [Gaussian() for ii in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].mean = v

    @property
    def covariances(self):
        """List of Gaussian covariances, each is a N x N numpy array. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "scale")

    @covariances.setter
    def covariances(self, val):
        if not isinstance(val, list):
            warn("Must set covariances to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for ii in range(len(val))]
            self._distributions = [Gaussian() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].covariance = v

    def add_components(self, means, covariances, weights):
        """Add Gaussian distributions to the mixture.

        Parameters
        ----------
        means : list
            Each is a N x 1 numpy array of the mean of the distributions to add.
        covariances : list
            Each is a N x N numpy array of the covariance of the distributions
            to add.
        weights : list
            Each is a float for the weight of the distributions to add. No
            normalization is done.

        Returns
        -------
        None.
        """
        if not isinstance(means, list):
            means = [
                means,
            ]
        if not isinstance(covariances, list):
            covariances = [
                covariances,
            ]
        if not isinstance(weights, list):
            weights = [
                weights,
            ]

        self._distributions.extend(
            [Gaussian(mean=m, covariance=c) for m, c in zip(means, covariances)]
        )
        self.weights.extend(weights)


class StudentsTMixture(BaseMixtureModel):
    """Students T mixture object."""

    def __init__(self, means=None, scalings=None, dof=None, **kwargs):
        if means is not None and scalings is not None and dof is not None:
            if isinstance(dof, list):
                dists = [
                    StudentsT(mean=m, scale=s, dof=df)
                    for m, s, df in zip(means, scalings, dof)
                ]
            else:
                dists = [
                    StudentsT(mean=m, scale=s, dof=dof) for m, s in zip(means, scalings)
                ]
            kwargs["distributions"] = dists
        super().__init__(**kwargs)

    @property
    def means(self):
        """List of Gaussian means, each is a N x 1 numpy array. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "location")

    @means.setter
    def means(self, val):
        if not isinstance(val, list):
            warn("Must set means to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for ii in range(len(val))]
            self._distributions = [StudentsT() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].mean = v

    @property
    def covariances(self):
        """Read only list of covariances, each is a N x N numpy array."""
        return _DistListWrapper(self._distributions, "covariance")

    @property
    def scalings(self):
        """List of scalings, each is a N x N numpy array. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "scale")

    @scalings.setter
    def scalings(self, val):
        if not isinstance(val, list):
            warn("Must set scalings to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for ii in range(len(val))]
            self._distributions = [StudentsT() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].scale = v

    @property
    def dof(self):
        """Most common degree of freedom for the mixture. Deprecated but kept for compatability, new code should use degrees_of_freedom."""
        vals, counts = np.unique(
            [d.degrees_of_freedom for d in self._distributions],
            return_counts=True,
        )
        inds = np.argwhere(counts == np.max(counts))
        return vals[inds[0]].item()

    @dof.setter
    def dof(self, val):
        for d in self._distributions:
            d.degrees_of_freedom = val

    @property
    def degrees_of_freedom(self):
        """List of degrees of freedom, each is a float. Recommended to be read only."""
        return _DistListWrapper(self._distributions, "degrees_of_freedom")

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, val):
        if not isinstance(val, list):
            warn("Must set degrees of freedom to a list")
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for ii in range(len(val))]
            self._distributions = [StudentsT() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].degrees_of_freedom = v

    def add_components(self, means, scalings, dof_lst, weights):
        """Add Student's t-distributions to the mixture.

        Parameters
        ----------
        means : list
            Each is a N x 1 numpy array of the mean of the distributions to add.
        scalings : list
            Each is a N x N numpy array of the scale of the distributions
            to add.
        dof_lst : list
            Each is a float representing the degrees of freedom of the distribution
            to add.
        weights : list
            Each is a float for the weight of the distributions to add. No
            normalization is done.

        Returns
        -------
        None.
        """
        if not isinstance(means, list):
            means = [
                means,
            ]
        if not isinstance(scalings, list):
            scalings = [
                scalings,
            ]
        if not isinstance(dof_lst, list):
            dof_lst = [
                dof_lst,
            ]
        if not isinstance(weights, list):
            weights = [
                weights,
            ]

        self._distributions.extend(
            [
                StudentsT(mean=m, scale=s, dof=df)
                for m, s, df in zip(means, scalings, dof_lst)
            ]
        )
        self.weights.extend(weights)


class SymmetricGaussianPareto(BaseSingleModel):
    """Represents a Symmetric Gaussian-Pareto Mixture Distribution object.

    Attributes
    ----------
    location : float
        Location (mean) of the distribution. By definition, it is always 0.
    scale : float
        Standard deviation of the core Gaussian region. The default is None.
    threshold : float
        Location where the core Gaussian region meets the Pareto tail. The default is None.
    tail_shape : float
        GPD shape parameter of the distribution's tail. The default is None
    tail_scale : float
        GPD scale parameter of the distribution's tail. The default is None

    """

    def __init__(
        self,
        location=0,
        scale=None,
        threshold=None,
        tail_shape=None,
        tail_scale=None,
    ):
        """Initialize an object.

        Parameters
        ----------
        location : float
            Location (mean) of the distribution. By definition, it is always 0.
        scale : float, optional
            Sets the class attribute "scale".
        threshold : float, optional
            Sets the class attribute "threshold".
        tail_shape : float, optional
            Sets the class attribute "tail_shape".
        tail_scale : float, optional
            Sets the class attribute "tail_scale".

        Returns
        -------
        None.
        """
        super().__init__(loc=np.zeros((1, 1)), scale=scale)
        self.threshold = threshold
        self.tail_shape = tail_shape
        self.tail_scale = tail_scale

    def sample(self, rng: rnd._generator = None, num_samples: int = None) -> np.ndarray:
        """Generate a random sample from the distribution model.

        Parameters
        ----------
        num_samples : int
            Specify the size of the sample.

        Returns
        -------
        N numpy array
            Numpy array containing a random sample of the specified size from
            the distribution.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1

        p = rng.uniform(size=num_samples)
        p_sorted = np.sort(p)
        F_mu = norm.cdf(-self.threshold, loc=self.location, scale=self.scale)
        F_u = norm.cdf(self.threshold, loc=self.location, scale=self.scale)

        idx_mu = int(np.argmax(p_sorted >= F_mu))
        idx_u = int(np.argmax(p_sorted >= F_u))

        sample = np.zeros((num_samples, self.location.size))
        transformed_left = np.add(np.negative(p_sorted[0:idx_mu]), 1)
        transformed_left = np.subtract(transformed_left, F_u)
        transformed_left = np.divide(transformed_left, (1 - F_u))
        left_samp = genpareto.ppf(
            transformed_left, self.tail_shape, loc=0, scale=self.tail_scale
        )
        left_samp = np.subtract(np.negative(left_samp), self.threshold)

        transformed_right = np.subtract(p_sorted[idx_u:], F_u)
        transformed_right = np.divide(transformed_right, (1 - F_u))
        right_samp = genpareto.ppf(
            transformed_right, self.tail_shape, loc=0, scale=self.tail_scale
        )
        right_samp = np.add(right_samp, self.threshold)

        center_samp = norm.ppf(
            p_sorted[idx_mu:idx_u], loc=self.location, scale=self.scale
        )

        sample[0:idx_mu] = np.transpose(left_samp)
        sample[idx_mu:idx_u] = np.transpose(center_samp)
        sample[idx_u:] = np.transpose(right_samp)

        return sample

    def CI(self, alfa):
        """Return confidence interval of distribution given a significance level 'alfa'.

        Parameters
        ----------
        alfa : float
            significance level, i.e. confidence level = (1 - alfa). Must be
            a positive real number which is less than 1

        Returns
        -------
        1 x 2 numpy array
            Numpy array containing the upper and lower bound of the computed
            confidence interval.
        """
        q_u = halfnorm.cdf(self.threshold, loc=self.location, scale=self.scale)
        q_x = 1 - alfa
        if q_x <= q_u:
            value = halfnorm.ppf(q_x, loc=self.location, scale=self.scale)
        else:
            temp = (q_x - q_u) / (1 - q_u)
            value = self.threshold + genpareto.ppf(
                temp, self.tail_shape, loc=0, scale=self.tail_scale
            )
        return np.array([[-value, value]])

    def CDFplot(self, data):
        """Plot the overbound and DKW bound(s) against ECDF of input data.

        Parameters
        ----------
        data : N numpy array
            Contains the error sample data for which the overbound was computed.

        Returns
        -------
        matplotlib line plot
            Shows empirical cumulative distribution function of input error
            data, the associated DKW bound(s), and the computed overbound in the
            CDF domain.
        """
        pos = np.absolute(data)
        sorted_abs_data = np.sort(pos)

        # Plot data ECDF, DKW lower bound, and Symmetric GPO in CDF domain

        n = data.size
        # confidence = 1 - 1e-6
        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))

        x_core = np.linspace(0, self.threshold, 10000)
        x_tail = np.linspace(self.threshold, 1.2 * max(sorted_abs_data), 10000)

        y_core = halfnorm.cdf(x_core, loc=self.location, scale=self.scale)
        y_tail = genpareto.cdf(
            x_tail - self.threshold,
            self.tail_shape,
            loc=self.location,
            scale=self.tail_scale,
        ) * (
            1 - (halfnorm.cdf(self.threshold, loc=self.location, scale=self.scale))
        ) + (
            halfnorm.cdf(self.threshold, loc=self.location, scale=self.scale)
        )

        x = np.append(x_core, x_tail)
        y = np.append(y_core, y_tail)

        ecdf_ords = np.zeros(n)

        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        DKW_lower_ords = np.subtract(ecdf_ords, epsilon)

        plt.figure("Symmetric GPO Plot in Half-Gaussian CDF Domain")
        plt.plot(sorted_abs_data, ecdf_ords, label="ECDF")
        plt.plot(sorted_abs_data, DKW_lower_ords, label="DKW Lower Bound")
        plt.plot(x, y, label="Symmetric GPO")

        plt.xlim([0, 1.2 * max(sorted_abs_data)])
        plt.legend()
        plt.grid()
        plt.title("Symmetric GPO Plot in CDF Domain")
        plt.ylabel("Accumulated Probability")
        plt.xlabel("Error Magnitude")

    def probscaleplot(self, data):
        """Generate probability plot of the ECDF, overbound, and DKW bound(s).

        Parameters
        ----------
        data : N numpy array
            numpy array containing the error data used to calculate the
            overbound

        Returns
        -------
        matplotlib line plot
            Shows empirical cumulative distribution function of input error
            data, the associated DKW bound(s), and the computed overbound
            in the CDF domain where the probability axis is represented with
            percentiles and is scaled such that a Gaussian CDF is linear.
        """
        pos = np.absolute(data)
        sorted_abs_data = np.sort(pos)
        n = data.size
        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))

        x_core = np.linspace(0, self.threshold, 10000)
        x_tail = np.linspace(self.threshold, max(sorted_abs_data), 10000)

        y_core = halfnorm.cdf(x_core, loc=self.location, scale=self.scale)
        y_tail = genpareto.cdf(
            x_tail - self.threshold,
            self.tail_shape,
            loc=self.location,
            scale=self.tail_scale,
        ) * (
            1 - (halfnorm.cdf(self.threshold, loc=self.location, scale=self.scale))
        ) + (
            halfnorm.cdf(self.threshold, loc=self.location, scale=self.scale)
        )
        x_ob = np.append(x_core, x_tail)
        y_ob = np.append(y_core, y_tail)
        dist_type = halfnorm()

        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n
        DKW_lower_ords = np.subtract(ecdf_ords, epsilon)

        x_ecdf = sorted_abs_data
        y_ecdf = ecdf_ords

        x_dkw = x_ecdf
        y_dkw = DKW_lower_ords

        figure, ax = plt.subplots()
        ax.set_ylim(bottom=5, top=99.999)
        ax.set_yscale("prob", dist=dist_type)

        plt.plot(x_ecdf, 100 * y_ecdf)
        plt.plot(x_dkw, 100 * y_dkw)
        plt.plot(x_ob, 100 * y_ob)
        plt.legend(["ECDF", "DKW Lower Bound", "Symmetric Gaussian-Pareto Overbound"])
        plt.title("Probability Plot of Symmetric Gaussian-Pareto Overbound")
        plt.ylabel("CDF Percentiles")
        plt.xlabel("Error Magnitude")
        plt.grid()


class PairedGaussianPareto(BaseSingleModel):
    """Represents a Paired Gaussian-Pareto Mixture Distribution object.

    Attributes
    ----------
    left_tail_shape : float
        GPD shape parameter (gamma) of the left tail. The default is None.
    left_tail_scale : float
        GPD scale parameter (beta) of the left tail. The default is None.
    left_threshold : float
        Location where the left tail meets the left Gaussian core region. The default is None.
    left_mean : float
        Mean of the left Gaussian core region. The default is None.
    left_sigma : float
        Standard deviation of the left Gaussian core region. The default is None.
    right_tail_shape : float
        GPD shape parameter (gamma) of the right tail. The default is None.
    right_tail_scale : float
        GPD scale parameter (beta) of the right tail. The default is None.
    right_threshold : float
        Location where the right tail meets the right Gaussian core region. The default is None.
    right_mean : float
        Mean of the right Gaussian core region. The default is None.
    right_sigma : float
        Standard deviation of the right Gaussian core region. The default is None.
    """

    def __init__(
        self,
        location=None,
        scale=None,
        left_tail_shape=None,
        left_tail_scale=None,
        left_threshold=None,
        left_mean=None,
        left_sigma=None,
        right_tail_shape=None,
        right_tail_scale=None,
        right_threshold=None,
        right_mean=None,
        right_sigma=None,
    ):
        """Initialize an object.

        Parameters
        ----------
        left_tail_shape : float
            Sets the class attribute of the same name
        left_tail_scale : float
            Sets the class attribute of the same name
        left_threshold : float
            Sets the class attribute of the same name
        left_mean : float
            Sets the class attribute of the same name
        left_sigma : float
            Sets the class attribute of the same name
        right_tail_shape : float
            Sets the class attribute of the same name
        right_tail_scale : float
            Sets the class attribute of the same name
        right_threshold : float
            Sets the class attribute of the same name
        right_mean : float
            Sets the class attribute of the same name
        right_sigma : float
            Sets the class attribute of the same name

        Returns
        -------
        None.
        """
        super().__init__(loc=np.zeros((1, 1)), scale=scale)
        self.left_tail_shape = left_tail_shape
        self.left_tail_scale = left_tail_scale
        self.left_threshold = left_threshold
        self.left_mean = left_mean
        self.left_sigma = left_sigma
        self.right_tail_shape = right_tail_shape
        self.right_tail_scale = right_tail_scale
        self.right_threshold = right_threshold
        self.right_mean = right_mean
        self.right_sigma = right_sigma

    def sample(self, rng: rnd._generator = None, num_samples: int = None) -> np.ndarray:
        """Generate a random sample from the distribution model.

        Parameters
        ----------
        num_samples : int
            Specify the size of the sample.

        Returns
        -------
        N numpy array
            Numpy array containing a random sample of the specified size from
            the distribution.
        """
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1

        p = rng.uniform(size=num_samples)
        p_sorted = np.sort(p)

        FuL = norm.cdf(self.left_threshold, loc=self.left_mean, scale=self.left_sigma)
        FuR = norm.cdf(
            self.right_threshold, loc=self.right_mean, scale=self.right_sigma
        )

        lt_ords = p_sorted[p_sorted < FuL]
        lc_ords = p_sorted[p_sorted >= FuL]
        lc_ords = lc_ords[lc_ords <= 0.5]
        rc_ords = p_sorted[p_sorted > 0.5]
        rc_ords = rc_ords[rc_ords <= FuR]
        rt_ords = p_sorted[p_sorted > FuR]

        lt_ords = np.divide(np.add(np.negative(lt_ords), FuL), FuL)
        lt_samp = np.transpose(
            genpareto.ppf(lt_ords, self.left_tail_shape, scale=self.left_tail_scale)
        )
        lt_samp = np.add(np.negative(lt_samp), self.left_threshold)

        rt_ords = np.divide(np.subtract(rt_ords, FuR), (1 - FuR))
        rt_samp = np.transpose(
            genpareto.ppf(rt_ords, self.right_tail_shape, scale=self.right_tail_scale)
        )
        rt_samp = np.add(rt_samp, self.right_threshold)

        lc_samp = np.transpose(
            norm.ppf(lc_ords, loc=self.left_mean, scale=self.left_sigma)
        )
        rc_samp = np.transpose(
            norm.ppf(rc_ords, loc=self.right_mean, scale=self.right_sigma)
        )

        samp = np.concatenate((lt_samp, lc_samp, rc_samp, rt_samp))

        return samp

    def CI(self, alfa):
        """Return confidence interval of distribution given a significance level 'alfa'.

        Parameters
        ----------
        alfa : float
            significance level, i.e. confidence level = (1 - alfa). Must be
            a positive real number which is less than 1

        Returns
        -------
        1 x 2 numpy array
            Numpy array containing the upper and lower bound of the computed
            confidence interval.
        """
        p = alfa / 2

        FuL = norm.cdf(self.left_threshold, loc=self.left_mean, scale=self.left_sigma)
        FuR = norm.cdf(
            self.right_threshold, loc=self.right_mean, scale=self.right_sigma
        )

        if p > FuL:
            left = norm.ppf(p, loc=self.left_mean, scale=self.left_sigma)
        else:
            p_Lt = (FuL - p) / (FuL)
            temp = genpareto.ppf(p_Lt, self.left_tail_shape, scale=self.left_tail_scale)
            left = -temp + self.left_threshold

        if (1 - p) < FuR:
            right = norm.ppf((1 - p), loc=self.right_mean, scale=self.right_sigma)
        else:
            val = ((1 - p) - FuR) / (1 - FuR)
            temp = genpareto.ppf(
                val, self.right_tail_shape, scale=self.right_tail_scale
            )
            right = temp + self.right_threshold

        return np.array([[left, right]])

    def CDFplot(self, data):
        """Plot the overbound and DKW bound(s) against ECDF of input data.

        Parameters
        ----------
        data : N numpy array
            Contains the error sample data for which the overbound was computed.

        Returns
        -------
        matplotlib line plot
            Shows empirical cumulative distribution function of input error
            data, the associated DKW bound(s), and the computed overbound in the
            CDF domain.
        """
        n = data.size
        data_sorted = np.sort(data)

        ecdf_ords = np.zeros(n)

        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))

        dkw_high = np.add(ecdf_ords, epsilon)
        dkw_low = np.subtract(ecdf_ords, epsilon)

        x_ob_left_tail = np.linspace(
            1.05 * data_sorted[0], self.left_threshold, num=1000
        )
        sub = np.flip(
            np.negative(np.subtract(x_ob_left_tail, self.left_threshold + 1e-14))
        )
        y_ob_left_tail = np.transpose(
            genpareto.cdf(
                sub,
                self.left_tail_shape,
                scale=self.left_tail_scale,
            )
        )

        Fu = norm.cdf(self.left_threshold, loc=self.left_mean, scale=self.left_sigma)
        y_ob_left_tail = np.flip(
            np.add(np.negative(np.multiply(y_ob_left_tail, Fu)), Fu)
        )

        x_ob_core = np.linspace(self.left_threshold, self.right_threshold, num=10000)
        y_ob_core = np.zeros(10000)

        for i in range(10000):
            if x_ob_core[i] >= self.left_threshold and x_ob_core[i] < self.left_mean:
                y_ob_core[i] = norm.cdf(
                    x_ob_core[i], loc=self.left_mean, scale=self.left_sigma
                )
            elif x_ob_core[i] >= self.left_mean and x_ob_core[i] <= self.right_mean:
                y_ob_core[i] = 0.5
            elif (
                x_ob_core[i] > self.right_mean and x_ob_core[i] <= self.right_threshold
            ):
                y_ob_core[i] = norm.cdf(
                    x_ob_core[i], loc=self.right_mean, scale=self.right_sigma
                )

        x_ob_right_tail = np.linspace(
            self.right_threshold, 1.05 * data_sorted[-1], num=1000
        )
        sub = np.subtract(x_ob_right_tail, self.right_threshold - 1e-14)
        y_ob_right_tail = np.transpose(
            genpareto.cdf(sub, self.right_tail_shape, scale=self.right_tail_scale)
        )

        Fu = norm.cdf(self.right_threshold, loc=self.right_mean, scale=self.right_sigma)
        y_ob_right_tail = np.add(np.multiply(y_ob_right_tail, (1 - Fu)), Fu)

        x_ob = np.concatenate((x_ob_left_tail, x_ob_core, x_ob_right_tail))
        y_ob = np.concatenate((y_ob_left_tail, y_ob_core, y_ob_right_tail))

        plt.figure("Paired GPO Plot in CDF Domain")
        plt.plot(data_sorted, ecdf_ords, label="ECDF")
        plt.plot(data_sorted, dkw_high, label="DKW Upper Bound")
        plt.plot(data_sorted, dkw_low, label="DKW Lower Bound")
        plt.plot(x_ob, y_ob, label="Paired Gaussian-Pareto Overbound")

        left_edge = 1.1 * x_ob_left_tail[0]
        right_edge = 1.1 * x_ob_right_tail[-1]
        plt.xlim([left_edge, right_edge])
        plt.legend()
        plt.grid()
        plt.title("Paired GPO Plot in CDF Domain")
        plt.ylabel("Accumulated Probability")
        plt.xlabel("Error")

    def probscaleplot(self, data):
        """Generate probability plot of the ECDF, overbound, and DKW bound(s).

        Parameters
        ----------
        data : N numpy array
            numpy array containing the error data used to calculate the
            overbound

        Returns
        -------
        matplotlib line plot
            Shows empirical cumulative distribution function of input error
            data, the associated DKW bound(s), and the computed overbound
            in the CDF domain where the probability axis is represented with
            percentiles and is scaled such that a Gaussian CDF is linear.
        """
        n = data.size
        data_sorted = np.sort(data)

        ecdf_ords = np.zeros(n)

        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))

        dkw_high = np.add(ecdf_ords, epsilon)
        dkw_low = np.subtract(ecdf_ords, epsilon)

        x_ob_left_tail = np.linspace(
            1.05 * data_sorted[0], self.left_threshold, num=1000
        )
        sub = np.flip(
            np.negative(np.subtract(x_ob_left_tail, self.left_threshold + 1e-14))
        )
        y_ob_left_tail = np.transpose(
            genpareto.cdf(
                sub,
                self.left_tail_shape,
                scale=self.left_tail_scale,
            )
        )

        Fu = norm.cdf(self.left_threshold, loc=self.left_mean, scale=self.left_sigma)
        y_ob_left_tail = np.flip(
            np.add(np.negative(np.multiply(y_ob_left_tail, Fu)), Fu)
        )

        x_ob_core = np.linspace(self.left_threshold, self.right_threshold, num=10000)
        y_ob_core = np.zeros(10000)

        for i in range(10000):
            if x_ob_core[i] >= self.left_threshold and x_ob_core[i] < self.left_mean:
                y_ob_core[i] = norm.cdf(
                    x_ob_core[i], loc=self.left_mean, scale=self.left_sigma
                )
            elif x_ob_core[i] >= self.left_mean and x_ob_core[i] <= self.right_mean:
                y_ob_core[i] = 0.5
            elif (
                x_ob_core[i] > self.right_mean and x_ob_core[i] <= self.right_threshold
            ):
                y_ob_core[i] = norm.cdf(
                    x_ob_core[i], loc=self.right_mean, scale=self.right_sigma
                )

        x_ob_right_tail = np.linspace(
            self.right_threshold, 1.05 * data_sorted[-1], num=1000
        )
        sub = np.subtract(x_ob_right_tail, self.right_threshold - 1e-14)
        y_ob_right_tail = np.transpose(
            genpareto.cdf(sub, self.right_tail_shape, scale=self.right_tail_scale)
        )

        Fu = norm.cdf(self.right_threshold, loc=self.right_mean, scale=self.right_sigma)
        y_ob_right_tail = np.add(np.multiply(y_ob_right_tail, (1 - Fu)), Fu)

        x_ob = np.concatenate((x_ob_left_tail, x_ob_core, x_ob_right_tail))
        y_ob = np.concatenate((y_ob_left_tail, y_ob_core, y_ob_right_tail))

        figure, ax = plt.subplots()
        ax.set_ylim(bottom=0.001, top=99.999)
        ax.set_yscale("prob")

        plt.plot(data_sorted, 100 * ecdf_ords)
        plt.plot(data_sorted, 100 * dkw_high)
        plt.plot(data_sorted, 100 * dkw_low)
        plt.plot(x_ob, 100 * y_ob)
        plt.legend(
            [
                "ECDF",
                "DKW Upper Bound",
                "DKW Lower Bound",
                "Paired Gaussian-Pareto Overbound",
            ]
        )
        plt.title("Probability Plot of Paired Gaussian-Pareto Overbound")
        plt.ylabel("CDF Percentiles")
        plt.xlabel("Error")
        plt.grid()
