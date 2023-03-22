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
    fig.axes[0].hist(
        z, density=True, cumulative=True, bins=1000, histtype="stepfilled"
    )
    fig.suptitle("Emperical CDF of Polynomial")
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
