# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:07:45 2022

@author: Vaughn Weirens
"""

import numpy as np
import serums.models
import serums.enums
import serums.distribution_estimator as de


def test_Gaussian_KS(num_samples=1000):
    print("Testing Gaussian distribution with Kolmogorov Smirnov Test:\n")
    dist = serums.models.Gaussian(mean=np.array([0]), covariance=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(dist,
                                               serums.enums.GoodnessOfFitTest.KOLMOGOROV_SMIRNOV,
                                               samples)
    fmt = "Input location : {}, Input Scale : {}\nOutput location : {}, Output Scale : {}\n"
    print(fmt.format(dist.location, dist.scale,
                     out_dist.location, out_dist.scale))


def test_Gaussian_CVM(num_samples=1000):
    print("Testing Gaussian distribution with Cramer von Mises Criterion:\n")
    dist = serums.models.Gaussian(mean=np.array([0]), covariance=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(dist,
                                               serums.enums.GoodnessOfFitTest.CRAMER_VON_MISES,
                                               samples)
    fmt = "Input location : {}, Input Scale : {}\nOutput location : {}, Output Scale : {}\n"
    print(fmt.format(dist.location, dist.scale,
                     out_dist.location, out_dist.scale))


def test_StudentsT_KS(num_samples=1000):
    print("Testing Student's T distribution with Kolmogorov Smirnov Test:\n")
    dist = serums.models.StudentsT(mean=np.array([0]), scale=np.array([[1]]),
                                   dof=np.array([30]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(dist,
                                               serums.enums.GoodnessOfFitTest.KOLMOGOROV_SMIRNOV,
                                               samples)
    fmt = "Input location : {}, Input Scale : {}, Input DOF : {}\nOutput location : {}, Output Scale : {}, Output DOF : {}\n"
    print(fmt.format(dist.location, dist.scale, dist.degrees_of_freedom,
                     out_dist.location, out_dist.scale,
                     out_dist.degrees_of_freedom))


def test_StudentsT_CVM(num_samples=1000):
    print("Testing Student's T distribution with Cramer von Mises Criterion:\n")
    dist = serums.models.StudentsT(mean=np.array([0]), scale=np.array([[1]]),
                                   dof=np.array([30]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(dist,
                                               serums.enums.GoodnessOfFitTest.CRAMER_VON_MISES,
                                               samples)
    fmt = "Input location : {}, Input Scale : {}, Input DOF : {}\nOutput location : {}, Output Scale : {}, Output DOF : {}\n"
    print(fmt.format(dist.location, dist.scale, dist.degrees_of_freedom,
                     out_dist.location, out_dist.scale,
                     out_dist.degrees_of_freedom))


def test_Cauchy_KS(num_samples=1000):
    print("Testing Cauchy distribution with Kolmogorov Smirnov Test")


def test_Cauchy_CVM(num_samples=1000):
    print("Testing Cauchy distribution with Cramer von Mises Criterion")


if __name__ == "__main__":
    test_Gaussian_KS()
    test_Gaussian_CVM()
    test_StudentsT_KS()
    test_StudentsT_CVM()
    # test_Cauchy_KS()
    # test_Cauchy_CVM()
