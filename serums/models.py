"""Defines various distribution models."""
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
from warnings import warn

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

    def __init__(self, loc=None, scale=None):
        super().__init__()
        self.location = loc
        self.scale = scale

    def sample(self, rng=None):
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
        warn('sample not implemented by class {}'.format(type(self).__name__))

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
        warn('pdf not implemented by class {}'.format(type(self).__name__))
        return np.nan


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
        return rng.multivariate_normal(self.mean.flatten(), self.covariance)

    def pdf(self, x):
        """Multi-variate probability density function for this distribution.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.multivariate_normal
        return rv.pdf(x.flatten(), mean=self.mean.flatten(), cov=self.covariance)


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
            msg = 'Degrees of freedom is {} and must be > 2'
            raise RuntimeError(msg.format(self._dof))
        return self._dof / (self._dof - 2) * self.scale

    @covariance.setter
    def covariance(self, val):
        warn('Covariance is read only.')

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
        return rv.pdf(x.flatten(), loc=self.location.flatten(), shape=self.scale,
                      df=self.degrees_of_freedom)

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
        x = rv.rvs(loc=self.location.flatten(),
                   shape=self.scale, df=self.degrees_of_freedom)

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
            msg = 'Degrees of freedom is {} and must be > 0'
            raise RuntimeError(msg.format(self._dof))
        return (self._dof * 2) * (self.scale**2)

    @covariance.setter
    def covariance(self, val):
        warn('Covariance is read only.')

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
        return rv.pdf(x.flatten(), self._dof,
                      loc=self.location.flatten(), shape=self.scale)

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
        x = rv.rvs(self._dof, loc=self.location.flatten(),
                   scale=self.scale)

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
        warn('Mean does not exist for a Cauchy')

    @mean.setter
    def mean(self, val):
        warn('Mean does not exist for a Cauchy')

    @property
    def degrees_of_freedom(self):
        """Degrees of freedom of the distribution, fixed at 1."""
        return super().degrees_of_freedom

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, value):
        warn('Degrees of freedom is 1 for a Cauchy')

    @property
    def covariance(self):
        """Read only covariance of the distribution (if defined)."""
        warn('Covariance is does not exist.')

    @covariance.setter
    def covariance(self, val):
        warn('Covariance is does not exist.')


class GaussianScaleMixture(BaseSingleModel):
    """Helper class for defining Gaussian Scale Mixture objects.

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

    def __init__(self, gsm_type, location=None, location_range=None,
                 scale=None, scale_range=None, degrees_of_freedom=None,
                 df_range=None):
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
            raise RuntimeError('Type ({}) must be a GSMType'.format(gsm_type))

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
            msg = 'GSM type {:s} does not have a degree of freedom.'.format(self.type)
            warn(msg)
            return None

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, val):
        if self.type in self.__df_types:
            if self.type is enums.GSMTypes.CAUCHY:
                warn('GSM type {:s} requires degree of freedom = 1'.format(self.type))
                return
            self._df = val
        else:
            msg = ('GSM type {:s} does not have a degree of freedom. '
                   + 'Skipping').format(self.type)
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
            raise RuntimeError('GSM type: {} is not supported'.format(self.type))

    def _sample_student_t(self, rng):
        return stats.t.rvs(self.degrees_of_freedom, scale=self.scale,
                           random_state=rng)

    def _sample_SaS(self, rng):
        raise RuntimeError('sampling SaS distribution not implemented')


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
        self._shape = shape

    @property
    def location(self):
        """Location of the distribution.

        Returns
        -------
        N x 1 numpy array.
        """
        return self.location

    @location.setter
    def location(self, val):
        self.location = val

    @property
    def scale(self):
        """Scale of the distribution.

        Returns
        -------
        N x 1 numpy array.
        """
        return self.scale

    @scale.setter
    def scale(self, val):
        self.scale = val

    @property
    def shape(self):
        """Shape of the distribution.

        Returns
        -------
        N x 1 numpy array.
        """
        return self.shape

    @shape.setter
    def shape(self, val):
        self.shape = val
    
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
        x = (self.scale * rv.rvs(self.shape) ) + self.location

        return x.reshape((x.size, 1))

    def pdf(self, x):
        """Multi-variate probability density function for this distribution.

        Returns
        -------
        float
            PDF value of the state `x`.
        """
        rv = stats.genpareto
        return rv.pdf( (x.flatten() - self.location ) / self.scale, shape=self.shape.flatten()) / self.scale

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
        mix_ind = rng.choice(np.arange(len(self.weights), dtype=int),
                             p=self.weights)
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
        for w, dist in zip(self.weights, self._distributions):
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

    def add_component(self, *args):
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
        warn('add_component not implemented by {}'.format(type(self).__name__))


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
            return [getattr(self.dist_lst[ii], self.attr)
                    for ii in range(index.start, index.stop, step)]
        elif isinstance(index, int):
            return getattr(self.dist_lst[index], self.attr)

        else:
            fmt = 'Index must be a integer or slice not {}'
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
            fmt = 'Index must be a integer or slice not {}'
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
        raise RuntimeError('Cannot append, use add_component function instead.')

    def extend(self, *args):
        raise RuntimeError('Cannot extend, use add_component function instead.')


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
            kwargs['distributions'] = [Gaussian(mean=m, covariance=c)
                                       for m, c in zip(means, covariances)]
        super().__init__(**kwargs)

    @property
    def means(self):
        """List of Gaussian means, each is a N x 1 numpy array. Recommended to be read only."""
        return _DistListWrapper(self._distributions, 'location')

    @means.setter
    def means(self, val):
        if not isinstance(val, list):
            warn('Must set means to a list')
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for ii in range(len(val))]
            self._distributions = [Gaussian() for ii in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].mean = v

    @property
    def covariances(self):
        """List of Gaussian covariances, each is a N x N numpy array. Recommended to be read only."""
        return _DistListWrapper(self._distributions, 'scale')

    @covariances.setter
    def covariances(self, val):
        if not isinstance(val, list):
            warn('Must set covariances to a list')
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
            means = [means, ]
        if not isinstance(covariances, list):
            covariances = [covariances, ]
        if not isinstance(weights, list):
            weights = [weights, ]

        self._distributions.extend([Gaussian(mean=m, covariance=c)
                                   for m, c in zip(means, covariances)])
        self.weights.extend(weights)


class StudentsTMixture(BaseMixtureModel):
    """Students T mixture object."""

    def __init__(self, means=None, scalings=None, dof=None, **kwargs):
        if means is not None and scalings is not None and dof is not None:
            if isinstance(dof, list):
                dists = [StudentsT(mean=m, scale=s, dof=df)
                         for m, s, df in zip(means, scalings, dof)]
            else:
                dists = [StudentsT(mean=m, scale=s, dof=dof)
                         for m, s in zip(means, scalings)]
            kwargs['distributions'] = dists
        super().__init__(**kwargs)

    @property
    def means(self):
        """List of Gaussian means, each is a N x 1 numpy array. Recommended to be read only."""
        return _DistListWrapper(self._distributions, 'location')

    @means.setter
    def means(self, val):
        if not isinstance(val, list):
            warn('Must set means to a list')
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for ii in range(len(val))]
            self._distributions = [StudentsT() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].mean = v

    @property
    def covariances(self):
        """Read only list of covariances, each is a N x N numpy array."""
        return _DistListWrapper(self._distributions, 'covariance')

    @property
    def scalings(self):
        """List of scalings, each is a N x N numpy array. Recommended to be read only."""
        return _DistListWrapper(self._distributions, 'scale')

    @scalings.setter
    def scalings(self, val):
        if not isinstance(val, list):
            warn('Must set scalings to a list')
            return

        if len(val) != len(self._distributions):
            self.weights = [1 / len(val) for ii in range(len(val))]
            self._distributions = [StudentsT() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].scale = v

    @property
    def dof(self):
        """Most common degree of freedom for the mixture. Deprecated but kept for compatability, new code should use degrees_of_freedom."""
        vals, counts = np.unique([d.degrees_of_freedom for d in self._distributions],
                                 return_counts=True)
        inds = np.argwhere(counts == np.max(counts))
        return vals[inds[0]].item()

    @dof.setter
    def dof(self, val):
        for d in self._distributions:
            d.degrees_of_freedom = val

    @property
    def degrees_of_freedom(self):
        """List of degrees of freedom, each is a float. Recommended to be read only."""
        return _DistListWrapper(self._distributions, 'degrees_of_freedom')

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, val):
        if not isinstance(val, list):
            warn('Must set degrees of freedom to a list')
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
            means = [means, ]
        if not isinstance(scalings, list):
            scalings = [scalings, ]
        if not isinstance(dof_lst, list):
            dof_lst = [dof_lst, ]
        if not isinstance(weights, list):
            weights = [weights, ]

        self._distributions.extend([StudentsT(mean=m, scale=s, dof=df)
                                   for m, s, df in zip(means, scalings, dof_lst)])
        self.weights.extend(weights)
