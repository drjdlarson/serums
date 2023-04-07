def main():
    import numpy as np
    import matplotlib.pyplot as plt

    from serums.distribution_overbounder import fusion
    from serums.models import Gaussian

    x = Gaussian(mean=np.array([[2]]), covariance=np.array([[4]]))
    y = Gaussian(mean=np.array([[-3]]), covariance=np.array([[6]]))
    x.monte_carlo_size = 1e5

    poly = lambda x_, y_: x_ + y_ + x_**2

    z = fusion([x, y], poly)

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].hist(z, density=True, cumulative=True, bins=1000, histtype="stepfilled")
    fig.suptitle("Emperical CDF of Polynomial")
    fig.tight_layout()

    return fig


def multiplication():
    import scipy.special
    import numpy as np
    import matplotlib.pyplot as plt

    from serums.distribution_overbounder import fusion
    from serums.models import Gaussian

    std_x = np.sqrt(4)
    std_y = np.sqrt(6)

    x = Gaussian(mean=np.array([[0]]), covariance=np.array([[std_x**2]]))
    y = Gaussian(mean=np.array([[0]]), covariance=np.array([[std_y**2]]))
    x.monte_carlo_size = 1e6

    poly = lambda x_, y_: x_ * y_

    z = fusion([x, y], poly)

    plt_x_bnds = (-2, 2)
    pts = np.arange(*plt_x_bnds, 0.01)
    true_pdf = scipy.special.kn(0, np.abs(pts) / (std_x * std_y)) / (
        np.pi * std_x * std_y
    )

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    fig.axes[0].hist(
        z,
        density=True,
        bins=int(1e4),
        histtype="stepfilled",
        label="Emperical",
    )
    fig.axes[0].plot(pts, true_pdf, label="True", color="k", zorder=1000)
    fig.axes[0].legend(loc="upper left")
    fig.suptitle("PDF of Polynomial")
    fig.axes[0].set_xlim(plt_x_bnds)
    fig.tight_layout()

    return fig


def run():
    import os

    print("Generating Gaussian Fusion example")

    fout = os.path.join(
        os.path.dirname(__file__), "{}.png".format(os.path.basename(__file__)[:-3])
    )
    if not os.path.isfile(fout):
        fig = main()
        fig.savefig(fout)

    fout = os.path.join(
        os.path.dirname(__file__),
        "{}_multiplication.png".format(os.path.basename(__file__)[:-3]),
    )
    if not os.path.isfile(fout):
        fig = multiplication()
        fig.savefig(fout)
