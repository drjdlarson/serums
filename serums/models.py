"""Defines various distribution models."""
from __future__ import annotations
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
from warnings import warn
import matplotlib.pyplot as plt
from scipy.stats import probplot
from copy import deepcopy

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
        self.location = loc
        self.scale = scale

        self.monte_carlo_size = monte_carlo_size

    def sample(
        self, rng: rnd._generator = None, num_samples: int = None
    ) -> np.ndarray:
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
                fmt.format(
                    self.mean.ravel()[ii], *self.covariance[ii, :].tolist()
                )
                + "\n"
            )
        return msg

    def __sub__(self, other: BaseSingleModel) -> np.ndarray:
        if self.location.size != other.location.size:
            raise RuntimeError(
                "Can not subtract distributions of different shapes ({:d} vs {:d})".format(
                    self.location.size, other.location.size
                )
            )
        return self.sample(num_samples=self.monte_carlo_size) - other.sample(
            num_samples=self.monte_carlo_size
        )

    def __neg__(self) -> np.ndarray:
        """Should only be used to redefine order of operations for a subtraction operation (i.e. -g1 + g2 vs g2 - g1)"""
        return -self.sample(num_samples=self.monte_carlo_size)

    def __add__(self, other: BaseSingleModel) -> np.ndarray:
        if self.location.size != other.location.size:
            raise RuntimeError(
                "Can not add distributions of different shapes ({:d} vs {:d})".format(
                    self.location.size, other.location.size
                )
            )
        return self.sample(num_samples=self.monte_carlo_size) - other.sample(
            num_samples=self.monte_carlo_size
        )

    def __mul__(self, other: BaseSingleModel) -> np.ndarray:
        if self.location.size != other.location.size:
            raise RuntimeError(
                "Can not multiply distributions of different shapes ({:d} vs {:d})".format(
                    self.location.size, other.location.size
                )
            )
        return self.sample(num_samples=self.monte_carlo_size) - other.sample(
            num_samples=self.monte_carlo_size
        )

    def __truediv__(self, other: BaseSingleModel) -> np.ndarray:
        if self.location.size != other.location.size:
            raise RuntimeError(
                "Can not divide distributions of different shapes ({:d} vs {:d})".format(
                    self.location.size, other.location.size
                )
            )
        return self.sample(num_samples=self.monte_carlo_size) / other.sample(
            num_samples=self.monte_carlo_size
        )

    def __floordiv__(self, other: BaseSingleModel) -> np.ndarray:
        if self.location.size != other.location.size:
            raise RuntimeError(
                "Can not floor divide distributions of different shapes ({:d} vs {:d})".format(
                    self.location.size, other.location.size
                )
            )
        return self.sample(num_samples=self.monte_carlo_size) // other.sample(
            num_samples=self.monte_carlo_size
        )

    def __pow__(self, power: int, modulo=None) -> np.ndarray:

        return self.sample(num_samples=self.monte_carlo_size).__pow__(
            power, modulo=modulo
        )


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
                fmt.format(
                    self.mean.ravel()[ii], *self.covariance[ii, :].tolist()
                )
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
            self.mean.flatten(), self.covariance, size=num_samples
        )

    def pdf(self, x):
        """Multi-variate probability density function for this distribution.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.multivariate_normal
        return rv.pdf(
            x.flatten(), mean=self.mean.flatten(), cov=self.covariance
        )

    def CI(self, alfa):
        """Determine confidence interval given significance level alfa.

        Parameters
        ----------
        alfa : float
            significance level, i.e. confidence level = (1 - alfa)

        Returns
        -------
        2 x 1 numpy array
            array containing upper and lower bound of confidence interval
        """
        low = stats.norm.ppf(alfa, loc=self.mean, scale=np.sqrt(self.scale))
        high = stats.norm.ppf(
            1 - alfa, loc=self.mean, scale=np.sqrt(self.scale)
        )
        return np.array([[low[0, 0], high[0, 0]]])

    def CDFplot(self, data):
        """Plot Gaussian overbound and DKW bound against ECDF of input data.

        Parameters
        ----------
        data : N numpy array
            numpy array containing the error data used to calculate the
            overbound

        Returns
        -------
        unsure
        """
        n = data.size
        ordered_abs_data = np.sort(np.abs(data))
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        X_ECDF = ordered_abs_data
        X_OB = ordered_abs_data
        Y_ECDF = ecdf_ords
        Y_OB = stats.halfnorm.cdf(X_OB, loc=0, scale=np.sqrt(self.scale[0, 0]))

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        plt.figure("Half-Gaussian CDF Domain")
        plt.plot(X_ECDF, Y_ECDF, label="Original ECDF")
        plt.plot(X_ECDF, np.add(Y_ECDF, epsilon), label="DKW Upper Band")
        plt.plot(X_ECDF, np.subtract(Y_ECDF, epsilon), label="DKW Lower Band")
        plt.plot(X_OB, Y_OB, label="Overbound CDF")
        plt.xlim(np.array([0, 1.1 * ordered_abs_data[-1]]))
        plt.ylim(np.array([0, 1]))
        plt.legend()
        plt.grid()

    def Qplot(self, data):
        """Plot Gaussian overbound and ECDF quantiles against Gaussian quantiles."""
        plt.figure("Probability Plot of Symmetric Gaussian Test Case")
        probplot(data, plot=plt)
        plt.grid()


class PairedGaussian(BaseSingleModel):
    def __init__(self, left: Gaussian, right: Gaussian):
        super().__init__()

        self.left_gaussian = deepcopy(left)
        self.right_gaussian = deepcopy(right)

    def sample(
        self, rng: rnd._generator = None, num_samples: int = None
    ) -> np.ndarray:
        if rng is None:
            rng = rnd.default_rng()
        if num_samples is None:
            num_samples = 1

        if rng.uniform() > 0.5:
            return self.right_gaussian.sample(rng=rng, num_samples=num_samples)
        else:
            return self.left_gaussian.sample(rng=rng, num_samples=num_samples)


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

    def sample(self, rng=None):
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

        rv = stats.multivariate_t
        rv.random_state = rng
        x = rv.rvs(
            loc=self.location.flatten(),
            shape=self.scale,
            df=self.degrees_of_freedom,
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

    def sample(self, rng=None):
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

        rv = stats.chi2
        rv.random_state = rng
        x = rv.rvs(self._dof, loc=self.location.flatten(), scale=self.scale)

        return x.reshape((x.size, 1))


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
        match the type of the :attr:`.location` attribute.
    scale_range : tuple
        Minimum and maximum values for the scale parameter. Useful if being
        fed to a filter for estimating the scale parameter. Each element must
        match the type of the :attr:`.scale` attribute. The default is None.
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
            match the type of the :attr:`.location` attribute. The default is None
        scale : N x N numpy array, optional
            Scale parameter of the distribution being represented as a GSM.
            The default is None.
        scale_range : tuple, optional
            Minimum and maximum values for the scale parameter. Useful if being
            fed to a filter for estimating the scale parameter. Each element must
            match the type of the :attr:`.scale` attribute. The default is None.
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
            msg = "GSM type {:s} does not have a degree of freedom.".format(
                self.type
            )
            warn(msg)
            return None

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, val):
        if self.type in self.__df_types:
            if self.type is enums.GSMTypes.CAUCHY:
                warn(
                    "GSM type {:s} requires degree of freedom = 1".format(
                        self.type
                    )
                )
                return
            self._df = val
        else:
            msg = (
                "GSM type {:s} does not have a degree of freedom. "
                + "Skipping"
            ).format(self.type)
            warn(msg)

    def sample(self, rng=None):
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

        if self.type in [enums.GSMTypes.STUDENTS_T, enums.GSMTypes.CAUCHY]:
            return self._sample_student_t(rng)

        elif self.type is enums.GSMTypes.SYMMETRIC_A_STABLE:
            return self._sample_SaS(rng)

        else:
            raise RuntimeError(
                "GSM type: {} is not supported".format(self.type)
            )

    def _sample_student_t(self, rng):
        return stats.t.rvs(
            self.degrees_of_freedom, scale=self.scale, random_state=rng
        )

    def _sample_SaS(self, rng):
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

    def sample(self, rng=None):
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

        rv = stats.genpareto
        rv.random_state = rng
        x = (self.scale * rv.rvs(self.shape)) + self.location

        return x.reshape((x.size, 1))

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

    def __init__(self, distributions=None, weights=None):
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

    def __iter__(self):
        """Allow iterating over mixture objects by (weight, distribution)."""
        return BaseMixtureModelIterator(self.weights, self._distributions)

    def __len__(self):
        """Give the number of distributions in the mixture."""
        return len(self._distributions)

    def sample(self, rng=None):
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
        mix_ind = rng.choice(np.arange(len(self), dtype=int), p=self.weights)
        x = self._distributions[mix_ind].sample(rng=rng)
        return x.reshape((x.size, 1))

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
        raise RuntimeError(
            "Cannot append, use add_component function instead."
        )

    def extend(self, *args):
        raise RuntimeError(
            "Cannot extend, use add_component function instead."
        )


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
                Gaussian(mean=m, covariance=c)
                for m, c in zip(means, covariances)
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
            [
                Gaussian(mean=m, covariance=c)
                for m, c in zip(means, covariances)
            ]
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
                    StudentsT(mean=m, scale=s, dof=dof)
                    for m, s in zip(means, scalings)
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
    """Represents a Symmetric Guassian-Pareto Mixture Distribution object.

    Attributes
    ----------
    location : 1 x 1 numpy array
        location of the distribution. By definition, it is always 0.
    scale : 1 x 1 numpy array
        variance of the core Gaussian region. The default is None.
    threshold : 1 x 1 numpy array
        location where the core Gaussian region meets the Pareto tail. The default is None.
    tail_shape : 1 x 1 numpy array
        GPD shape parameter of the distribution's tail. The default is None
    tail_scale : 1 x 1 numpy array
        GPD scale parameter of the distribution's tail. The default is None

    """

    def __init__(
        self,
        scale=None,
        threshold=None,
        tail_shape=None,
        tail_scale=None,
    ):
        """Initialize an object.

        Parameters
        ----------
        scale : 1 x 1 numpy array, optional
            variance of the core Gaussian region. The default is None.
        threshold : 1 x 1 numpy array, optional
            location where the core Gaussian region meets the Pareto tail. The default is None.
        tail_shape : 1 x 1 numpy array, optional
            GPD shape parameter of the distribution's tail. The default is None
        tail_scale : 1 x 1 numpy array, optional
            GPD scale parameter of the distribution's tail. The default is None

        Returns
        -------
        None.
        """
        super().__init__(loc=np.zeros((1, 1)), scale=scale)
        self.threshold = threshold
        self.tail_shape = tail_shape
        self.tail_scale = tail_scale

    def confidence_interval(self, alfa):
        """Return confidence interval of distribution given significance level.

        Parameters
        ----------
        alfa : float
            significance level of returned confidence interval

        Returns
        -------
        2 numpy array
            the interval containing the largest deviations from 0 that will happen with
            probability P = (1-alfa).
        """
        pass

    def plot_ECDF_and_OB_in_CDF_domain(self):
        """Plots the input data ECDF against the computed overbound CDF.

        Parameters
        ----------
        None.

        Returns
        -------
        matplotlib line plot
            shows empirical distribution function of input error data, the
            associated DKW Lower bound, and the computed symmetric GPO in the
            CDF domain.
        """
        pass
