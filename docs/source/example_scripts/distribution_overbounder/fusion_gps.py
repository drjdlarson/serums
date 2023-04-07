def main():
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import inv

    from serums.distribution_overbounder import fusion
    from serums.models import Gaussian

    # assume that user position and gps SV positions are known
    user_pos = np.array([-41.772709, -16.789194, 6370.059559, 999.76252931])
    gps_pos_lst = np.array(
        [
            [15600, 7540, 20140],
            [18760, 2750, 18610],
            [17610, 14630, 13480],
            [19170, 610, 18390],
        ],
        dtype=float,
    )

    # define the noise on each pseudorange as a Gaussian (assume the same covariance for simplicity)
    std_pr = 10
    pr_means = np.sqrt(np.sum((gps_pos_lst - user_pos[:3]) ** 2, axis=1))
    prs = []
    for m in pr_means:
        prs.append(
            Gaussian(
                mean=m.reshape((-1, 1)).copy(), covariance=np.array([[std_pr**2]])
            )
        )

    # define the uncertainty matrix
    R = np.diag(std_pr * np.ones(len(prs)))

    # create the linearized GPS measurement matrix with a function
    def make_G(user_pos_: np.ndarray) -> np.ndarray:
        G = None
        for ii, sv_pos in enumerate(gps_pos_lst):
            diff = user_pos_.ravel()[:3] - sv_pos
            row = np.hstack((diff / np.sqrt(np.sum(diff**2)), np.array([1])))
            if G is None:
                G = row
            else:
                G = np.vstack((G, row))
        return G

    # define the mapping matrix (this will be used in the polynomial)
    G = make_G(user_pos)
    S = inv(G.T @ R @ G) @ G.T @ inv(R)

    fig = plt.figure()
    [fig.add_subplot(2, 2, ii + 1) for ii in range(4)]
    fig.suptitle("Emperical CDF of GPS Outputs")
    ttl_lst = ["X Pos", "Y Pos", "Z Pos", "Time Delay"]

    # define the fusion polynomial for each output variable (ie x/y/z pos and time delay)
    for ii, ttl in enumerate(ttl_lst):
        poly = (
            lambda pr0, pr1, pr2, pr3: S[ii, 0] * pr0
            + S[ii, 1] * pr1
            + S[ii, 2] * pr2
            + S[ii, 3] * pr3
        )
        # these samples could then be run through an overbounding routine
        samples = fusion(prs, poly)

        # plot the emperical CDF to view results
        fig.axes[ii].hist(
            samples,
            density=True,
            cumulative=True,
            bins=int(1e3),
            histtype="stepfilled",
        )
        fig.axes[ii].set_title(ttl)
        if ii == 2 or ii == 3:
            fig.axes[ii].set_xlabel("meters")

    fig.tight_layout()

    return fig


def run():
    import os

    print("Generating GPS Fusion example")

    fout = os.path.join(
        os.path.dirname(__file__), "{}.png".format(os.path.basename(__file__)[:-3])
    )
    if not os.path.isfile(fout):
        fig = main()
        fig.savefig(fout)
