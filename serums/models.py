"""For defining various distribution models."""
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
from warnings import warn

import serums.enums as enums


class BaseSingleModel:
    def __init__(self, loc=None, scale=None):
        self.location = loc
        self.scale = scale

    def sample(self, rng=None):
        warn('sample not implemented by class {}'.format(type(self).__name__))

    def pdf(self, x):
        warn('pdf not implemented by class {}'.format(type(self).__name__))
        return np.nan


class Gaussian(BaseSingleModel):
    def __init__(self, mean=None, covar=None):
        super().__init__(loc=mean, scale=covar)

    @property
    def mean(self):
        return self.location

    @mean.setter
    def mean(self, val):
        self.location = val

    @property
    def covariance(self):
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
    def __init__(self, mean=None, scale=None, dof=None):
        super().__init__(loc=mean, scale=scale)
        self.degrees_of_freedom = dof

    @property
    def mean(self):
        return self.location

    @mean.setter
    def mean(self, val):
        self.location = val

    @property
    def covariance(self):
        if self.dof <= 2:
            msg = 'Degrees of freedom is {} and must be > 2'
            raise RuntimeError(msg.format(self.dof))
        return self.dof / (self.dof - 2) * self.scale

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
        return rv.pdf(x.flatten(), loc=self.mean.flatten(), shape=self.scale,
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
        x = rv.rvs(loc=self.mean.flatten(),
                   shape=self.scale, df=self.degrees_of_freedom)

        return x.reshape((x.size, 1))


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
    location_range : tuple, optional
        Minimum and maximum values for the location parameter. Useful if being
        fed to a filter for estimating the location parameter. Each element must
        match the type of the :attr:`.location` attribute.
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
        location : TYPE, optional
            DESCRIPTION. The default is None.
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


class BaseMixtureModel:
    def __init__(self,  distributions=None, weights=None):
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


class _DistListWrapper:
    def __init__(self, dist_lst, attr):
        self.dist_lst = dist_lst
        self.attr = attr

    def __getitem__(self, index):
        if isinstance(index, slice):
            step = 1
            if index.step is not None:
                step = index.step
            return [getattr(self.dist_lst[ii], self.attr)
                    for ii in range(index.start, index.stop, step)]
        elif isinstance(index, int):
            return getattr(self.dist_lst[index], self.attr)

        else:
            raise RuntimeError('Index must be a integer or slice')

    def __setitem__(self, index, val):
        if isinstance(index, slice):
            step = 1
            if index.step is not None:
                step = index.step
            for ii in range(index.start, index.stop, step):
                setattr(self.dist_lst[ii], self.attr, val)

        elif isinstance(index, int):
            setattr(self.dist_lst[index], self.attr, val)

        else:
            raise RuntimeError('Index must be a integer or slice')

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
        if means is not None and covariances is not None:
            kwargs['distributions'] = [Gaussian(mean=m, covar=c)
                                       for m, c in zip(means, covariances)]
        super().__init__(**kwargs)

    @property
    def means(self):
        """List of Gaussian means, each is a N x 1 numpy array."""
        return _DistListWrapper(self._distributions, 'location')

    @means.setter
    def means(self, val):
        if not isinstance(val, list):
            warn('Must set means to a list')
            return

        if len(val) != len(self._distributions):
            self._distributions = [Gaussian() for ii in range(len(val))]
        for ii, v in enumerate(val):
            self._distributions[ii].mean = v

    @property
    def covariances(self):
        """List of Gaussian covariances, each is a N x N numpy array."""
        return _DistListWrapper(self._distributions, 'scale')

    @covariances.setter
    def covariances(self, val):
        if not isinstance(val, list):
            warn('Must set covariances to a list')
            return

        if len(val) != len(self._distributions):
            self._distributions = [Gaussian() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].covariance = v

    def add_components(self, means, covariances, weights):
        self._distributions.extend([Gaussian(mean=m, covar=c)
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
    def covariances(self):
        """List of Gaussian covariances, each is a N x N numpy array."""
        return _DistListWrapper(self._distributions, 'covariance')

    @property
    def scalings(self):
        return _DistListWrapper(self._distributions, 'scale')

    @scalings.setter
    def scalings(self, val):
        if not isinstance(val, list):
            warn('Must set scalings to a list')
            return

        if len(val) != len(self._distributions):
            self._distributions = [StudentsT() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].scale = v

    @property
    def dof(self):
        vals, counts = np.unique([d.degrees_of_freedom for d in self._distributions],
                                 return_counts=True)
        inds = np.argwhere(counts == np.max(counts))
        return vals[inds[0]]

    @dof.setter
    def dof(self, val):
        for d in self._distributions:
            d.degrees_of_freedom = val

    @property
    def degrees_of_freedom(self):
        return _DistListWrapper(self._distributions, 'degrees_of_freedom')

    @degrees_of_freedom.setter
    def degrees_of_freedom(self, val):
        if not isinstance(val, list):
            warn('Must set degrees of freedom to a list')
            return

        if len(val) != len(self._distributions):
            self._distributions = [StudentsT() for ii in range(len(val))]

        for ii, v in enumerate(val):
            self._distributions[ii].degrees_of_freedom = v
