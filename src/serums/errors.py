"""Defines custom exceptions."""


class DistributionEstimatorFailed(Exception):
    """Thrown when the distribution estimator fails."""

    pass


class OverboundingMethodFailed(Exception):
    """Thrown when the chosen overbounding method is not applicable for the given input error data."""

    pass
