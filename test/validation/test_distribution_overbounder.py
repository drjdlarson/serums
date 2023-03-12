"""For testing the different overbound types."""

import numpy as np
import matplotlib.pyplot as plt
import serums.distribution_overbounder as sdob
import serums.models as smodels
from scipy.stats import norm, genpareto, t, halfnorm, probplot
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
        alfa = 1e-6
        interval = out_dist.CI(alfa)
        print(
            "Confidence Interval = (",
            interval[0, 0],
            ", ",
            interval[0, 1],
            ")",
        )
        out_dist.CDFplot(data)
        out_dist.Qplot(data)


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
        # confidence = 1 - 1e-6
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
    print("Testing Paired Gaussian Overbounder:")
    np.random.seed(seed=233423)
    n = 10000
    left = norm.rvs(loc=0, scale=1.5, size=int(n / 2))
    left = np.abs(left)
    left = -left
    left = np.sort(left)
    np.random.seed(seed=233423)
    right = norm.rvs(loc=0, scale=1, size=int(n / 2))
    right = np.abs(right)
    right = right
    right = np.sort(right)
    data = np.concatenate((left, right))

    paired_gaussian_overbounder = sdob.PairedGaussianOverbounder()
    PairedOB_object = paired_gaussian_overbounder.overbound(
        data, debug_plots=DEBUG
    )

    if DEBUG:
        ecdf_ords = np.zeros(n)
        for i in range(n):
            ecdf_ords[i] = (i + 1) / n

        confidence = 0.95
        alfa = 1 - confidence
        epsilon = np.sqrt(np.log(2 / alfa) / (2 * n))
        DKW_low = np.subtract(ecdf_ords, epsilon)
        DKW_high = np.add(ecdf_ords, epsilon)

        left_mean = PairedOB_object.left_gaussian.mean
        left_std = np.sqrt(PairedOB_object.left_gaussian.covariance)
        right_mean = PairedOB_object.right_gaussian.mean
        right_std = np.sqrt(PairedOB_object.right_gaussian.covariance)

        y_left_ob = np.reshape(
            norm.cdf(data, loc=left_mean, scale=left_std), (n,)
        )
        y_right_ob = np.reshape(
            norm.cdf(data, loc=right_mean, scale=right_std), (n,)
        )
        x_paired_ob = np.linspace(
            np.min(data) - 1, np.max(data) + 1, num=10000
        )
        y_paired_ob = np.zeros(x_paired_ob.size)
        left_pt = PairedOB_object.left_gaussian.mean
        right_pt = PairedOB_object.right_gaussian.mean

        for i in range(y_paired_ob.size):
            if x_paired_ob[i] < left_pt:
                y_paired_ob[i] = norm.cdf(
                    x_paired_ob[i], loc=left_mean, scale=left_std
                )
            elif x_paired_ob[i] > right_pt:
                y_paired_ob[i] = norm.cdf(
                    x_paired_ob[i], loc=right_mean, scale=right_std
                )
            else:
                y_paired_ob[i] = 0.5

        plt.figure("Paired Overbound in CDF Domain")
        plt.plot(data, y_left_ob, label="Left OB", linestyle="--")
        plt.plot(data, y_right_ob, label="Right OB", linestyle="--")
        plt.plot(x_paired_ob, y_paired_ob, label="Paired OB")
        plt.plot(data, ecdf_ords, label="ECDF")
        plt.plot(data, DKW_high, label="Upper DKW Bound")
        plt.plot(data, DKW_low, label="Lower DKW Bound")
        plt.legend()

    pass


def test_PairedGPO():
    pass


if __name__ == "__main__":
    DEBUG = True
    # test_SymmetricGaussianOverbound()
    # test_SymmetricGPO()
    test_PairedGaussianOverbound()
    # test_PairedGPO()
