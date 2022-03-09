"""Enumerations used by SERUMS."""
import enum


@enum.unique
class GSMTypes(enum.Enum):
    """Gaussian Scale Mixture Types."""

    STUDENTS_T = enum.auto()
    """Student's t-distribution."""

    CAUCHY = enum.auto()
    """Cauchy distribution."""

    SYMMETRIC_A_STABLE = enum.auto()
    r"""Symmetric :math:`\alpha`-stable distribution."""

    def __str__(self):
        """Return the enum name for strings."""
        return self.name


@enum.unique
class SingleObjectDistance(enum.Enum):
    """Enumeration for distance methods."""

    MANHATTAN = enum.auto()
    r"""The Manhattan/taxicab/:math:`L_1` distance.

    Notes
    -----
    Uses the form

    .. math::
        d(x, y) = \Sigma_i \vert x_i - y_i \vert
    """

    EUCLIDEAN = enum.auto()
    r"""The euclidean distance between two points.

    Notes
    -----
    Uses the form :math:`d(x, y) = \sqrt{(x-y)^T(x-y)}`.
    """

    HELLINGER = enum.auto()
    r"""The hellinger distance between two probability distributions.

    Notes
    -----
    It is at most 1, and for Gaussian distributions it takes the form

    .. math::
        d_H(f,g) &= 1 - \sqrt{\frac{\sqrt{\det{\left[\Sigma_x \Sigma_y\right]}}}
                              {\det{\left[0.5\Sigma\right]}}} \exp{\epsilon} \\
        \epsilon &= \frac{1}{4}(x - y)^T\Sigma^{-1}(x - y) \\
        \Sigma &= \Sigma_x + \Sigma_y
    """

    MAHALANOBIS = enum.auto()
    r"""The Mahalanobis distance between a point and a distribution.

    Notes
    -----
    Uses the form :math:`d(x, y) = \sqrt{(x-y)^T\Sigma_y^{-1}(x-y)}`.
    """

    def __str__(self):
        """Return the enum name for strings."""
        return self.name


@enum.unique
class MultiObjectDistance(enum.Enum):
    """Enumeration of multi-object distance types."""

    OSPA = enum.auto()
    """The Optimal Sub-Pattern Assignment distance between two point processes."""

    OSPA2 = enum.auto()
    """The Optimal Sub-Pattern Assignment(2)` distance."""

    def __str__(self):
        """Return the enum name for strings."""
        return self.name


@enum.unique
class GoodnessOfFitTest(enum.Enum):
    """Enumeration of Goodness of Fit Tests for distribution parameter estimation."""

    CRAMER_VON_MISES = enum.auto()
    """Cramer von Mises criterion for goodness of fit of a distribution to a set of samples."""

    KOLMOGOROV_SMIRNOV = enum.auto()
    """Kolmogorov Smirnov test for goodness of fit of a distribution to a set of samples."""

    ANDERSON_DARLING = enum.auto()
    """Anderson=Darling test for goodness of fit of two sets of samples.
    This test puts higher emphasis on the tails of the distribution."""

    def __str__(self):
        """Return the enum name for strings."""
        return self.name
