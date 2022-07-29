# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:07:45 2022

@author: Vaughn Weirens
"""

import numpy as np
import serums.models as smodels
from serums.enums import GoodnessOfFitTest
import serums.distribution_estimator as de


def test_Gaussian_KS(num_samples=1000):
    print("Testing Gaussian distribution with Kolmogorov Smirnov Test:")
    dist = smodels.Gaussian(mean=np.array([0]), covariance=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.KOLMOGOROV_SMIRNOV, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}\n\tOutput location : {}, Output Scale : {}\n"
    print(
        fmt.format(
            dist.location, dist.scale, out_dist.location, out_dist.scale
        )
    )


def test_Gaussian_CVM(num_samples=1000):
    print("Testing Gaussian distribution with Cramer von Mises Criterion:")
    dist = smodels.Gaussian(mean=np.array([0]), covariance=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.CRAMER_VON_MISES, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}\n\tOutput location : {}, Output Scale : {}\n"
    print(
        fmt.format(
            dist.location, dist.scale, out_dist.location, out_dist.scale
        )
    )


def test_Gaussian_AD(num_samples=1000):
    print("Testing Gaussian distribution with Anderson-Darling Test:")
    dist = smodels.Gaussian(mean=np.array([0]), covariance=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.ANDERSON_DARLING, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}\n\tOutput location : {}, Output Scale : {}\n"
    print(
        fmt.format(
            dist.location, dist.scale, out_dist.location, out_dist.scale
        )
    )


def test_StudentsT_KS(num_samples=5000):
    print("Testing Student's T distribution with Kolmogorov Smirnov Test:")
    dist = smodels.StudentsT(
        mean=np.array([0]), scale=np.array([[1]]), dof=np.array([30])
    )

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.KOLMOGOROV_SMIRNOV, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}, Input DOF : {}\n\tOutput location : {}, Output Scale : {}, Output DOF : {}\n"
    print(
        fmt.format(
            dist.location,
            dist.scale,
            dist.degrees_of_freedom,
            out_dist.location,
            out_dist.scale,
            out_dist.degrees_of_freedom,
        )
    )


def test_StudentsT_CVM(num_samples=5000):
    print("Testing Student's T distribution with Cramer von Mises Criterion:")
    dist = smodels.StudentsT(
        mean=np.array([0]), scale=np.array([[1]]), dof=np.array([30])
    )

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.CRAMER_VON_MISES, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}, Input DOF : {}\n\tOutput location : {}, Output Scale : {}, Output DOF : {}\n"
    print(
        fmt.format(
            dist.location,
            dist.scale,
            dist.degrees_of_freedom,
            out_dist.location,
            out_dist.scale,
            out_dist.degrees_of_freedom,
        )
    )


def test_StudentsT_AD(num_samples=1000):
    print("Testing Student's T distribution with Anderson-Darling Test:")
    dist = smodels.StudentsT(
        mean=np.array([0]), scale=np.array([[1]]), dof=np.array([30])
    )

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.ANDERSON_DARLING, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}, Input DOF : {}\n\tOutput location : {}, Output Scale : {}, Output DOF : {}\n"
    print(
        fmt.format(
            dist.location,
            dist.scale,
            dist.degrees_of_freedom,
            out_dist.location,
            out_dist.scale,
            out_dist.degrees_of_freedom,
        )
    )


def test_Cauchy_KS(num_samples=10000):
    print("Testing Cauchy distribution with Kolmogorov Smirnov Test:")
    dist = smodels.Cauchy(location=np.array([0]), scale=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.KOLMOGOROV_SMIRNOV, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}\n\tOutput location : {}, Output Scale : {}\n"
    print(
        fmt.format(
            dist.location, dist.scale, out_dist.location, out_dist.scale
        )
    )


def test_Cauchy_CVM(num_samples=10000):
    print("Testing Cauchy distribution with Cramer von Mises Criterion:")
    dist = smodels.Cauchy(location=np.array([0]), scale=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.CRAMER_VON_MISES, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}\n\tOutput location : {}, Output Scale : {}\n"
    print(
        fmt.format(
            dist.location, dist.scale, out_dist.location, out_dist.scale
        )
    )


def test_Cauchy_AD(num_samples=10000):
    print("Testing Cauchy distribution with Anderson-Darling Test:")
    dist = smodels.Cauchy(location=np.array([0]), scale=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, GoodnessOfFitTest.ANDERSON_DARLING, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}\n\tOutput location : {}, Output Scale : {}\n"
    print(
        fmt.format(
            dist.location, dist.scale, out_dist.location, out_dist.scale
        )
    )


def test_Grimshaw_MLE(num_samples=10000):
    print("Testing Grimshaw MLE:")
    # hardcode R code estimates
    shape_gamma = -0.5
    scale_beta = 2
    random_set = genpareto.rvs(shape_gamma, scale=scale_beta, size=1000)


if __name__ == "__main__":
    test_Gaussian_KS()
    test_Gaussian_CVM()
    test_Gaussian_AD()
    test_StudentsT_KS()
    test_StudentsT_CVM()
    test_StudentsT_AD()
    test_Cauchy_KS()
    test_Cauchy_CVM()
    test_Cauchy_AD()
