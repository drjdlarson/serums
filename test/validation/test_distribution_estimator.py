# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:07:45 2022

@author: Vaughn Weirens
"""

import numpy as np
import serums.models as smodels
from serums.enums import DistEstimatorMethod
import serums.distribution_estimator as de
from scipy.stats import genpareto
import matplotlib.pyplot as plt


def test_Gaussian_KS(num_samples=1000):
    print("Testing Gaussian distribution with Kolmogorov Smirnov Test:")
    dist = smodels.Gaussian(mean=np.array([0]), covariance=np.array([[1]]))

    samples = np.array([])
    for ii in range(0, num_samples):
        samples = np.append(samples, dist.sample())

    out_dist = de.estimate_distribution_params(
        dist, DistEstimatorMethod.KOLMOGOROV_SMIRNOV, samples
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
        dist, DistEstimatorMethod.CRAMER_VON_MISES, samples
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
        dist, DistEstimatorMethod.ANDERSON_DARLING, samples
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
        dist, DistEstimatorMethod.KOLMOGOROV_SMIRNOV, samples
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
        dist, DistEstimatorMethod.CRAMER_VON_MISES, samples
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
        dist, DistEstimatorMethod.ANDERSON_DARLING, samples
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
        dist, DistEstimatorMethod.KOLMOGOROV_SMIRNOV, samples
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
        dist, DistEstimatorMethod.CRAMER_VON_MISES, samples
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
        dist, DistEstimatorMethod.ANDERSON_DARLING, samples
    )
    fmt = "\tInput location : {}, Input Scale : {}\n\tOutput location : {}, Output Scale : {}\n"
    print(
        fmt.format(
            dist.location, dist.scale, out_dist.location, out_dist.scale
        )
    )


def test_GPD_Grimshaw_MLE(num_samples=1000):
    print("Testing Generalized Pareto Distribution with Grimshaw MLE: \n")
    Rshape = 0.4871232
    Rscale = 1.837633
    shape_gamma = 0.5
    scale_beta = 2

    dist = smodels.GeneralizedPareto(
        location=np.array([0]), scale=scale_beta, shape=shape_gamma
    )

    np.random.seed(seed=233423)
    random_set = genpareto.rvs(shape_gamma, scale=scale_beta, size=1000)
    out_dist = de.estimate_distribution_params(
        dist, DistEstimatorMethod.GRIMSHAW_MLE, random_set
    )
    fmt = "\tInput location : {}, Input Shape : {}, Input Scale : {}\n\tOutput location : {}, Output Shape : {}, Output Scale : {}\n\tR location : {}, R Shape : {}, R Scale : {}\n"
    print(
        fmt.format(
            dist.location,
            dist.shape,
            dist.scale,
            out_dist.location,
            out_dist.shape,
            out_dist.scale,
            "None",
            Rshape,
            Rscale,
        )
    )

    domain = np.linspace(0, 100, 1000)
    TEST_PDF = genpareto.pdf(domain, shape_gamma, 0, scale=scale_beta)

    fig, ax = plt.subplots(1, 1)
    ax.plot(domain, TEST_PDF)
    ax.hist(
        random_set,
        density=True,
        bins=np.linspace(0, 30, 150),
        histtype="stepfilled",
    )

    plt.xlim([0, 30])
    plt.show()


def test_GPD_MOM(num_samples=1000):
    print("Testing Generalized Pareto Distribution with MoM: \n")

    Rshape = -0.4393175
    Rscale = 1.8305034
    shape_gamma = -0.5
    scale_beta = 2

    dist = smodels.GeneralizedPareto(
        location=np.array([0]), scale=scale_beta, shape=shape_gamma
    )

    np.random.seed(seed=233423)
    random_set = genpareto.rvs(-0.5, scale=2, size=1000)
    out_dist = de.estimate_distribution_params(
        dist, DistEstimatorMethod.METHOD_OF_MOMENTS, random_set
    )
    fmt = "\tInput location : {}, Input Shape : {}, Input Scale : {}\n\tOutput location : {}, Output Shape : {}, Output Scale : {}\n\tR location : {}, R Shape : {}, R Scale : {}\n"
    print(
        fmt.format(
            dist.location,
            dist.shape,
            dist.scale,
            out_dist.location,
            out_dist.shape,
            out_dist.scale,
            "None",
            Rshape,
            Rscale,
        )
    )


def test_GPD_PWM(num_samples=1000):
    print("Testing Generalized Pareto Distribution with method of PWM: \n")

    Rshape = -0.4316918
    Rscale = 1.8208052
    shape_gamma = -0.5
    scale_beta = 2

    dist = smodels.GeneralizedPareto(
        location=np.array([0]), scale=scale_beta, shape=shape_gamma
    )

    np.random.seed(seed=233423)
    random_set = genpareto.rvs(-0.5, scale=2, size=1000)
    out_dist = de.estimate_distribution_params(
        dist, DistEstimatorMethod.PROBABILITY_WEIGHTED_MOMENTS, random_set
    )
    fmt = "\tInput location : {}, Input Shape : {}, Input Scale : {}\n\tOutput location : {}, Output Shape : {}, Output Scale : {}\n\tR location : {}, R Shape : {}, R Scale : {}\n"
    print(
        fmt.format(
            dist.location,
            dist.shape,
            dist.scale,
            out_dist.location,
            out_dist.shape,
            out_dist.scale,
            "None",
            Rshape,
            Rscale,
        )
    )


if __name__ == "__main__":
    # test_Gaussian_KS()
    # test_Gaussian_CVM()
    # test_Gaussian_AD()
    # test_StudentsT_KS()
    # test_StudentsT_CVM()
    # test_StudentsT_AD()
    # test_Cauchy_KS()
    # test_Cauchy_CVM()
    # test_Cauchy_AD()
    test_GPD_Grimshaw_MLE()
    test_GPD_MOM()
    test_GPD_PWM()
