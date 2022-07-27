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
    def autobin(self):
        # Check for autocorrelation in error distribution
        if np.any(np.corrcoef(self.error, self.dependencies) > 0.1):
            # Bisect error distribution
            self.bins = self.bisect(self.error)
        return self.bins

    # Function to bisect error distribution into 2 bins and store the result in binned
    def bisect(self, error):
        # Initialize binned error distribution
        bins = np.zeros(2)
        # Find the midpoint of the error distribution
        midpoint = np.mean(error)
        # Find the indices of the error distribution that are less than the midpoint
        indices = np.where(error < midpoint)[0]
        # Store the number of indices less than the midpoint in the first bin
        bins[0] = len(indices)
        # Store the number of indices greater than the midpoint in the second bin
        bins[1] = len(error) - len(indices)
        # Return the binned error distribution
        return bins
