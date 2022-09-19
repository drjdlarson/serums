"""For testing the different overbound types."""

import numpy as np
import matplotlib.pyplot as plt
import serums.distribution_overbounder as sdob
import serums.models as smodels
from scipy.stats import norm, genpareto


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
    pass


def test_PairedGaussianOverbound():
    pass


def test_PairedGPO():
    pass


if __name__ == "__main__":
    test_SymmetricGaussianOverbound()
    # test_SymmetricGPO()
    # test_PairedGaussianOverbound()
    # test_PairedGPO()
    plt.show()
