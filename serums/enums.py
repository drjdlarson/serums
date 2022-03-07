"""Enumerations used by SERUMS."""
import enum


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
