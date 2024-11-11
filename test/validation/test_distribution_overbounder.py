"""For testing the different overbound types."""

import numpy as np
import matplotlib.pyplot as plt
import serums.distribution_overbounder as sdob
import serums.models as smodels
from scipy.stats import norm, genpareto, t, halfnorm, probplot, chi2
import time
import pytest


DEBUG = False


def test_FusionGaussian():
    x = smodels.Gaussian(mean=np.array([[2]]), covariance=np.array([[4]]))
    y = smodels.Gaussian(mean=np.array([[-3]]), covariance=np.array([[6]]))
    x.monte_carlo_size = 1e5

    poly = lambda x_, y_: x_ + y_ + x_**2

    z = sdob.fusion([x, y], poly)

    if DEBUG:
        fig = plt.figure()
        fig.add_subplot(1, 1, 1)
        fig.axes[0].hist(
            z, density=True, cumulative=True, bins=1000, histtype="stepfilled"
        )
        fig.suptitle("Empirical CDF of Polynomial")


@pytest.mark.slow
def test_SymmetricGaussianOverbound():
    print("Testing Symmetric Gaussian Overbounder: \n")
    np.random.seed(seed=233423)
    data = norm.rvs(loc=0, scale=1, size=10000)
    mu_in = 0
    sigma_in = 1
    mu_out = 0
    sigma_out = 1.4279351025441724

    ob_std = sdob.SymmetricGaussianOverbounder()
    out_dist = ob_std.overbound(data)

    fmt = "\tInput Location : {}, Input Scale : {}\n\tExpected Output Location : {}, Expected Output Scale : {}, \n\tActual Output Location : {}, Actual Output Scale : {}\n"
    print(
        fmt.format(
            mu_in,
            sigma_in,
            mu_out,
            sigma_out,
            out_dist.location[0, 0],
            np.sqrt(out_dist.covariance[0, 0]),
        )
    )

    if DEBUG:
        alfa = 1e-6
        interval = out_dist.CI(alfa)
        print(
            "Confidence Interval for alfa = 1e-6 : (",
            interval[0, 0],
            ", ",
            interval[0, 1],
            ")",
        )
        out_dist.CDFplot(data)
        out_dist.probscaleplot(data)


@pytest.mark.slow
def test_SymmetricGPO():
    print("Testing Symmetric Gaussian-Pareto Overbounder:")
    np.random.seed(seed=233423)
    data = t.rvs(8, loc=0, scale=1, size=10000)
    tail_shape_exp = 0.15465916616848446
    tail_scale_exp = 3.634927976810183
    u_exp = 2.747164551208518
    core_sigma_exp = 1.3401913542555797

    GPOBer = sdob.SymmetricGPO()

    start = time.time()
    out_dist = GPOBer.overbound(data)
    end = time.time()

    print("\n------")
    print("Runtime for n = 10,000: ", round((end - start) / 60, 2), " minutes\n")

    fmt = "\tExpected Output Gamma : {}, Expected Output Beta : {}, Expected Output u : {}, Expected Output Core Sigma : {}, \n\tActual Output Gamma : {}, Actual Output Beta : {}, Actual Output u : {}, Actual Output Core Sigma : {}\n"
    print(
        fmt.format(
            tail_shape_exp,
            tail_scale_exp,
            u_exp,
            core_sigma_exp,
            out_dist.tail_shape,
            out_dist.tail_scale,
            out_dist.threshold,
            float(out_dist.scale),
        )
    )

    if DEBUG:
        test_sample = out_dist.sample(num_samples=10000)
        plt.figure()
        plt.title("ECDF of OB Model Sample Obtained through .sample Method")
        plt.hist(test_sample, bins=1000, cumulative=True, histtype="step")
        plt.grid()
        alfa = 1e-6
        interval = out_dist.CI(alfa)
        print(
            "Confidence Interval for alfa = 1e-6 : (",
            interval[0, 0],
            ", ",
            interval[0, 1],
            ")",
        )
        out_dist.CDFplot(data)
        out_dist.probscaleplot(data)
    pass


# @pytest.mark.slow
# def test_PairedGaussianOverbound():
#     print("Testing Paired Gaussian Overbounder:")
#     np.random.seed(seed=233423)
#     n = 10000
#     left = norm.rvs(loc=0, scale=1.5, size=int(n / 2))
#     left = np.abs(left)
#     left = -left
#     left = np.sort(left)
#     np.random.seed(seed=233423)
#     right = norm.rvs(loc=0, scale=1, size=int(n / 2))
#     right = np.abs(right)
#     right = right
#     right = np.sort(right)
#     data = np.concatenate((left, right))

#     paired_gaussian_overbounder = sdob.PairedGaussianOverbounder()
#     out_dist = paired_gaussian_overbounder.overbound(data, debug_plots=DEBUG)

#     if DEBUG:
#         test_sample = out_dist.sample(num_samples=10000)
#         plt.figure()
#         plt.title("ECDF of OB Model Sample Obtained through .sample Method")
#         plt.hist(test_sample, bins=1000, cumulative=True, histtype="step")
#         plt.grid()
#         alfa = 1e-6
#         interval = out_dist.CI(alfa)
#         print(
#             "Confidence Interval for alfa = 1e-6 : (",
#             interval[0, 0],
#             ", ",
#             interval[0, 1],
#             ")",
#         )
#         out_dist.CDFplot(data)
#         out_dist.probscaleplot(data)

#     pass


# @pytest.mark.slow
# def test_PairedGPO():
#     print("Testing Paired Gaussian-Pareto Overbounder:")
#     np.random.seed(seed=233423)
#     n = 10000
#     data = t.rvs(8, loc=0, scale=1, size=n)

#     pairedGPOBer = sdob.PairedGPO()

#     out_dist = pairedGPOBer.overbound(data)

#     if DEBUG:
#         test_sample = out_dist.sample(num_samples=10000)
#         sorted_sample = np.sort(test_sample)
#         ecdf_ords = np.zeros(sorted_sample.size)
#         for i in range(sorted_sample.size):
#             ecdf_ords[i] = (i + 1) / n
#         plt.figure("Paired GPO verify ().sample")
#         plt.title("ECDF of OB Model Sample Obtained through .sample Method")
#         plt.plot(sorted_sample, ecdf_ords, label="Computer Generated Sample")
#         plt.grid()
#         alfa = 1e-6
#         interval = out_dist.CI(alfa)
#         print(
#             "Confidence Interval for alfa = 1e-6 : (",
#             interval[0, 0],
#             ", ",
#             interval[0, 1],
#             ")",
#         )
#         out_dist.CDFplot(data)
#         out_dist.probscaleplot(data)

#     pass


if __name__ == "__main__":
    plt.close("all")
    DEBUG = True
    # test_FusionGaussian()
    # test_SymmetricGaussianOverbound()
    # test_SymmetricGPO()
    # test_PairedGaussianOverbound()
    # test_PairedGPO()

    plt.show()
