"""For preprocessing errors."""
import numpy as np
from scipy.stats import ks_2samp, cramervonmises_2samp, anderson_ksamp
import serums.models
import pandas as pd


class BinnedError:
    """Class to represent binned error distribution.
    This class must be given truth data, measured data, dependency arrays, an I/O mapping function, and preferences.

    Attributes
    ----------
    truth : array_like
        Truth data.
    measured : array_like
        Measured data.
    dependencies : array_like
        Dependency arrays.
    io_map : function
        I/O mapping function.
    preferences : dict
        Preferences. 
    """

    # Initialize binned error distribution
    def __init__(self, truth, measured, dependencies, io_map, preferences):
        self.truth = truth
        self.measured = measured
        self.dependencies = dependencies
        self.io_map = io_map
        self.preferences = preferences

    # Function to normalize error distribution by subtracting the truth from the measured data
    def normalize(self):
        self.error = self.measured - self.truth
        return self.error

    # Function to remove obvious outliers from error distribution by removing errors more than 10 standard deviations from the mean
    def remove_outliers(self):
        self.error = self.error[np.abs(self.error) < 10 * np.std(self.error)]
        return self.error

    # Function to recursively bin error distribution by checking for autocorrelation in the error distribution with the dependency array, and bisecting the error distribution if there is autocorrelation
    def autobin(self, error=None):
        # Check for autocorrelation in error distribution
        if not error:
            error = self.error
        if np.any(np.corrcoef(error, self.dependencies) > 0.1):
            # Bisect error distribution, and recursively call autobin on each bin
            bins = self.bisect(error)
            for bin in bins:
                self.autobin(bin)
        else:
            # Store error distribution as a list of bins
            self.bins = [error]
        return self.bins

    # Function to bisect error distribution into 2 bins at its midpoint and store the result as a list of bins
    def bisect(self, error):
        midpoint = int(len(error) / 2)
        bins = [error[:midpoint], error[midpoint:]]
        return bins
