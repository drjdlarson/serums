"""For testing the different overbound types."""

import numpy as np
import matplotlib.pyplot as plt
import serums.distribution_overbounder as sdob
import serums.models as smodels
from scipy.stats import norm, genpareto, t, halfnorm
import time


DEBUG = False


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
        n = data.size
        ordered_data = np.sort(data)
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        X_ECDF = ordered_data
        X_OB = ordered_data
        Y_ECDF = ecdf_ords
        Y_OB = norm.cdf(X_OB, loc=0, scale=np.sqrt(out_dist.scale[0, 0]))

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        plt.figure("Plot of Symmetric Gaussian Test Case")
        plt.plot(X_ECDF, Y_ECDF, label="Original ECDF")
        plt.plot(X_ECDF, np.add(Y_ECDF, epsilon), label="DKW Upper Band")
        plt.plot(X_ECDF, np.subtract(Y_ECDF, epsilon), label="DKW Lower Band")
        plt.plot(X_OB, Y_OB, label="Overbound CDF")
        plt.xlim(np.array([-4, 4]))
        plt.ylim(np.array([0, 1]))
        plt.legend()
        plt.grid()
        # Note: the overbound passes slightly under the DKW band because the plot
        # is in the Gaussian CDF domain, not the Half-Gaussian CDF domain


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
    (
        tail_shape,
        tail_scale,
        u,
        core_sigma,
    ) = GPOBer.overbound(data)
    end = time.time()

    print("\n------")
    print(
        "Runtime for n = 10,000: ", round((end - start) / 60, 2), " minutes\n"
    )

    fmt = "\tExpected Output Gamma : {}, Expected Output Beta : {}, Expected Output u : {}, Expected Output Core Sigma : {}, \n\tActual Output Gamma : {}, Actual Output Beta : {}, Actual Output u : {}, Actual Output Core Sigma : {}\n"
    print(
        fmt.format(
            tail_shape_exp,
            tail_scale_exp,
            u_exp,
            core_sigma_exp,
            tail_shape,
            tail_scale,
            u,
            core_sigma,
        )
    )

    if DEBUG:
        pos = np.absolute(data)
        sorted_abs_data = np.sort(pos)

        # Plot data ECDF, DKW lower bound, and Symmetric GPO in CDF domain

        n = data.size
        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))

        x_core = np.linspace(0, u, 10000)
        x_tail = np.linspace(u, 1.2 * max(sorted_abs_data), 10000)

        y_core = halfnorm.cdf(x_core, loc=0, scale=core_sigma)
        y_tail = genpareto.cdf(
            x_tail - u, tail_shape, loc=0, scale=tail_scale
        ) * (1 - (halfnorm.cdf(u, loc=0, scale=core_sigma))) + (
            halfnorm.cdf(u, loc=0, scale=core_sigma)
        )

        x = np.append(x_core, x_tail)
        y = np.append(y_core, y_tail)

        ecdf_ords = np.zeros(n)

        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        DKW_lower_ords = np.subtract(ecdf_ords, epsilon)

        # Calculate Gaussian OB for comparison
        gaussian_ober = sdob.SymmetricGaussianOverbounder()
        out_dist = gaussian_ober.overbound(data)

        X_OB = np.linspace(0, 1.2 * max(sorted_abs_data), 10000)
        Y_OB = halfnorm.cdf(X_OB, loc=0, scale=np.sqrt(out_dist.scale[0, 0]))

        plt.figure()
        plt.plot(sorted_abs_data, ecdf_ords, label="ECDF")
        plt.plot(sorted_abs_data, DKW_lower_ords, label="DKW Lower Bound")
        plt.plot(x, y, label="Symmetric GPO")
        plt.plot(X_OB, Y_OB, label="Symmetric Gaussian Overbound")

        plt.xlim([0, 1.2 * max(sorted_abs_data)])
        plt.legend()
    pass


def test_PairedGaussianOverbound():
    pass


def test_PairedGPO():
    pass


if __name__ == "__main__":
    DEBUG = True
    # test_SymmetricGaussianOverbound()
    test_SymmetricGPO()
    # test_PairedGaussianOverbound()
    # test_PairedGPO()
