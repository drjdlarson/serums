"""Functions for calculating distances."""
import numpy as np
import numpy.linalg as la
from scipy.optimize import linear_sum_assignment

import warnings

from serums.enums import SingleObjectDistance


def calculate_hellinger(x, y, cov_x, cov_y):
    r"""Calculates the hellinger distance between two Gaussian distributions.

    Notes
    -----
    It is at most 1, and for Gaussian distributions it takes the form

    .. math::
        d_H(f,g) &= 1 - \sqrt{\frac{\sqrt{\det{\left[\Sigma_x \Sigma_y\right]}}}
                              {\det{\left[0.5\Sigma\right]}}} \exp{\epsilon} \\
        \epsilon &= \frac{1}{4}(x - y)^T\Sigma^{-1}(x - y) \\
        \Sigma &= \Sigma_x + \Sigma_y

    Parameters
    ----------
    x : N x 1 numpy array
        first distribution location parameter.
    y : N x 1 numpy array
        Second distribution location parameter.
    cov_x : N x N numpy array
        First distribution covariance.
    cov_y : N x N numpy array
        Second distribution covariance.

    Returns
    -------
    float
        Hellinger distance.

    """
    cov = cov_x + cov_y
    diff = (x - y).reshape((cov.shape[0], 1))
    epsilon = -0.25 * diff.T @ la.inv(cov) @ diff

    return (1 - np.sqrt(np.sqrt(la.det(cov_x @ cov_y)) / la.det(0.5 * cov))
            * np.exp(epsilon))


def calculate_mahalanobis(x, y, cov):
    r"""Calculates the Mahalanobis distance between a point and a distribution.

    Notes
    -----
    Uses the form

    .. math::
        d(x, y) = \sqrt{(x-y)^T\Sigma_y^{-1}(x-y)}

    Parameters
    ----------
    x : N x 1 numpy array
        Point.
    y : N x 1 numpy array
        Distribution location parameter.
    cov : N x N numpy array
        Distribution covariance.

    Returns
    -------
    float
        Mahalanobis distance.
    """
    diff = (x - y).reshape((cov.shape[0], 1))
    return np.sqrt(diff.T @ cov @ diff)


def calculate_ospa(est_mat, true_mat, c, p, use_empty=True, core_method=None,
                   true_cov_mat=None, est_cov_mat=None):
    """Calculates the OSPA distance between the truth at all timesteps.

    Notes
    -----
    This calculates the Optimal SubPattern Assignment metric for the
    extracted states and the supplied truth point distributions. The
    calculation is based on
    :cite:`Schuhmacher2008_AConsistentMetricforPerformanceEvaluationofMultiObjectFilters`
    with much of the math defined in
    :cite:`Schuhmacher2008_ANewMetricbetweenDistributionsofPointProcesses`.
    A value is calculated for each timestep available in the data. This can
    use different distance metrics as the core distance. The default follows
    the main paper where the euclidean distance is used. Other options
    include the Hellinger distance
    (see :cite:`Nagappa2011_IncorporatingTrackUncertaintyintotheOSPAMetric`),
    or the Mahalanobis distance.

    Parameters
    ----------
    est_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`est_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to estimated states.
    true_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`true_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to the true states.
    c : float
        Distance cutoff for considering a point properly assigned. This
        influences how cardinality errors are penalized. For :math:`p = 1`
        it is the penalty given false point estimate.
    p : int
        The power of the distance term. Higher values penalize outliers
        more.
    use_empty : bool, Optional
        Flag indicating if empty values should be set to 0 or nan. The default
        of True is fine for most cases.
    core_method : :class:`.enums.SingleObjectDistance`, Optional
        The main distance measure to use for the localization component.
        The default value of None implies :attr:`.enums.SingleObjectDistance.EUCLIDEAN`.
    true_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`true_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the true states, the object order must be consistent
        with the truth matrix, and is only needed for core methods
        :attr:`.enums.SingleObjectDistance.HELLINGER`. The default value is None.
    est_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`est_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the estimated states, the object order must be consistent
        with the estimated matrix, and is only needed for core methods
        :attr:`.enums.SingleObjectDistance.MAHALANOBIS`. The default value is None.

    Returns
    -------
    ospa : numpy array
        OSPA values at each timestep.
    localization : numpy array
        Localization component of the OSPA value at each timestep.
    cardinality : numpy array
        Cardinality component of the OSPA value at each timestep.
    core_method : :class:`.enums.SingleObjectDistance`
        Method to use as the core distance statistic.
    c : float
        Maximum distance value used.
    p : int
        Power of the distance term used.
    distances : Ne x Nt x T numpy array
        Numpy array of distances, rows are estimated objects columns are truth.
    e_exists : Ne x T numpy array
        Bools indicating if the estimated object exists at that time.
    t_exists : Nt x T numpy array
        Bools indicating if the true object exists at that time.
    """
    # error checking on optional input arguments
    if core_method is None:
        core_method = SingleObjectDistance.EUCLIDEAN

    elif core_method is SingleObjectDistance.MAHALANOBIS and est_cov_mat is None:
        msg = 'Must give estimated covariances to calculate {:s} OSPA. Using {:s} instead'
        warnings.warn(msg.format(core_method, SingleObjectDistance.EUCLIDEAN))
        core_method = SingleObjectDistance.EUCLIDEAN

    elif core_method is SingleObjectDistance.HELLINGER and true_cov_mat is None:
        msg = 'Must save covariances to calculate {:s} OSPA. Using {:s} instead'
        warnings.warn(msg.format(core_method, SingleObjectDistance.EUCLIDEAN))
        core_method = SingleObjectDistance.EUCLIDEAN

    if core_method is SingleObjectDistance.HELLINGER:
        c = np.min([1, c]).item()

    # setup data structuers
    t_exists = np.logical_not(np.isnan(true_mat[0, :, :])).T
    e_exists = np.logical_not(np.isnan(est_mat[0, :, :])).T

    # compute distance for all permutations
    num_timesteps = true_mat.shape[1]
    nt_objs = true_mat.shape[2]
    ne_objs = est_mat.shape[2]
    distances = np.nan * np.ones((ne_objs, nt_objs, num_timesteps))
    comb = np.array(np.meshgrid(np.arange(ne_objs, dtype=int),
                                np.arange(nt_objs, dtype=int))).T.reshape(-1, 2)
    e_inds = comb[:, 0]
    t_inds = comb[:, 1]
    shape = (ne_objs, nt_objs)

    localization = np.nan * np.ones(num_timesteps)
    cardinality = np.nan * np.ones(num_timesteps)

    for tt in range(num_timesteps):
        # use proper core method
        if core_method is SingleObjectDistance.EUCLIDEAN:
            distances[:, :, tt] = np.sqrt(np.sum((true_mat[:, tt, t_inds]
                                                  - est_mat[:, tt, e_inds])**2,
                                                 axis=0)).reshape(shape)

        elif core_method is SingleObjectDistance.MANHATTAN:
            distances[:, :, tt] = np.sum(np.abs(true_mat[:, tt, t_inds]
                                                - est_mat[:, tt, e_inds]),
                                         axis=0).reshape(shape)

        elif core_method is SingleObjectDistance.HELLINGER:
            for row, col in zip(e_inds, t_inds):
                if not (e_exists[row, tt] and t_exists[col, tt]):
                    continue

                distances[row, col, tt] = calculate_hellinger(est_mat[:, tt, row],
                                                              true_mat[:, tt, col],
                                                              est_cov_mat[:, :, tt, row],
                                                              true_cov_mat[:, :, tt, col])

        elif core_method is SingleObjectDistance.MAHALANOBIS:
            for row, col in zip(e_inds, t_inds):
                if not (e_exists[row, tt] and t_exists[col, tt]):
                    continue

                distances[row, col, tt] = calculate_mahalanobis(est_mat[:, tt, row],
                                                                true_mat[:, tt, col],
                                                                est_cov_mat[:, :, tt, row])

        else:
            warnings.warn('Single-object distance {} is not implemented. SKIPPING'.format(core_method))
            core_method = None
            break

        # check for mismatch
        one_exist = np.logical_xor(e_exists[:, [tt]], t_exists[:, [tt]].T)
        empty = np.logical_and(np.logical_not(e_exists[:, [tt]]),
                               np.logical_not(t_exists[:, [tt]]).T)

        distances[one_exist, tt] = c
        if use_empty:
            distances[empty, tt] = 0
        else:
            distances[empty, tt] = np.nan

        distances[:, :, tt] = np.minimum(distances[:, :, tt], c)

        m = np.sum(e_exists[:, tt])
        n = np.sum(t_exists[:, tt])
        if n.astype(int) == 0 and m.astype(int) == 0:
            localization[tt] = 0
            cardinality[tt] = 0
            continue

        if n.astype(int) == 0 or m.astype(int) == 0:
            localization[tt] = 0
            cardinality[tt] = c
            continue

        cont_sub = distances[e_exists[:, tt], :, tt][:, t_exists[:, tt]]**p
        row_ind, col_ind = linear_sum_assignment(cont_sub)
        cost = cont_sub[row_ind, col_ind].sum()

        inv_max_card = 1. / np.max([n, m])
        card_diff = np.abs(n - m)
        inv_p = 1. / p
        c_p = c**p
        localization[tt] = (inv_max_card * cost)**inv_p
        cardinality[tt] = (inv_max_card * c_p * card_diff)**inv_p

    ospa = localization + cardinality

    return (ospa, localization, cardinality, core_method, c, p,
            distances, e_exists, t_exists)


def calculate_ospa2(est_mat, true_mat, c, p, win_len,
                    core_method=SingleObjectDistance.MANHATTAN, true_cov_mat=None,
                    est_cov_mat=None):
    """Calculates the OSPA(2) distance between the truth at all timesteps.

    Notes
    -----
    This calculates the OSPA-on-OSPA, or OSPA(2) metric as defined by
    :cite:`Beard2017_OSPA2UsingtheOSPAMetrictoEvaluateMultiTargetTrackingPerformance`
    and further explained in :cite:`Beard2020_ASolutionforLargeScaleMultiObjectTracking`.
    It can be thought of as the time averaged per track error between the true
    and estimated tracks. The inner OSPA calculation can use any suitable OSPA
    distance metric from :func:`.calculate_ospa`

    Parameters
    ----------
    est_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`est_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to estimated states.
    true_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`true_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to the true states.
    c : float
        Distance cutoff for considering a point properly assigned. This
        influences how cardinality errors are penalized. For :math:`p = 1`
        it is the penalty given false point estimate.
    p : int
        The power of the distance term. Higher values penalize outliers
        more.
    win_len : int
        Number of timesteps to average the OSPA over.
    core_method : :class:`.enums.SingleObjectDistance`, Optional
        The main distance measure to use for the localization component of the
        inner OSPA calculation.
        The default value of None implies :attr:`.enums.SingleObjectDistance.EUCLIDEAN`.
    true_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`true_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the true states, the object order must be consistent
        with the truth matrix, and is only needed for core methods
        :attr:`.enums.SingleObjectDistance.HELLINGER`. The default value is None.
    est_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`est_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the estimated states, the object order must be consistent
        with the estimated matrix, and is only needed for core methods
        :attr:`.enums.SingleObjectDistance.MAHALANOBIS`. The default value is None.

    Returns
    -------
    ospa2 : numpy array
        OSPA values at each timestep.
    localization : numpy array
        Localization component of the OSPA value at each timestep.
    cardinality : numpy array
        Cardinality component of the OSPA value at each timestep.
    core_method : :class:`.enums.SingleObjectDistance`
        Method to use as the core distance statistic.
    c : float
        Maximum distance value used.
    p : int
        Power of the distance term used.
    win_len : int
        Window length used.
    """
    # Note p is redundant here so set = 1
    (core_method, c, _,
     distances, e_exists, t_exists) = calculate_ospa(est_mat, true_mat, c, 1,
                                                     use_empty=False,
                                                     core_method=core_method,
                                                     true_cov_mat=true_cov_mat,
                                                     est_cov_mat=est_cov_mat)[3:9]

    num_timesteps = distances.shape[2]
    inv_p = 1. / p
    c_p = c**p

    localization = np.nan * np.ones(num_timesteps)
    cardinality = np.nan * np.ones(num_timesteps)

    for tt in range(num_timesteps):
        win_idx = np.array([ii for ii in range(max(tt - win_len + 1, 0),
                                               tt + 1)],
                           dtype=int)

        # find matrix of time averaged OSPA between tracks
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='Mean of empty slice')
            track_dist = np.nanmean(distances[:, :, win_idx], axis=2)

        track_dist[np.isnan(track_dist)] = 0

        valid_rows = np.any(e_exists[:, win_idx], axis=1)
        valid_cols = np.any(t_exists[:, win_idx], axis=1)
        m = np.sum(valid_rows)
        n = np.sum(valid_cols)

        if n.astype(int) <= 0 and m.astype(int) <= 0:
            localization[tt] = 0
            cardinality[tt] = 0
            continue

        if n.astype(int) <= 0 or m.astype(int) <= 0:
            cost = 0
        else:
            inds = np.logical_and(valid_rows.reshape((valid_rows.size, 1)),
                                  valid_cols.reshape((1, valid_cols.size)))
            track_dist = (track_dist[inds]**p).reshape((m.astype(int),
                                                        n.astype(int)))
            row_ind, col_ind = linear_sum_assignment(track_dist)
            cost = track_dist[row_ind, col_ind].sum()

        max_nm = np.max([n, m])
        localization[tt] = (cost / max_nm)**inv_p
        cardinality[tt] = (c_p * np.abs(m - n) / max_nm)**inv_p

    ospa2 = localization + cardinality

    return ospa2, localization, cardinality, core_method, c, p, win_len


# TODO: Rewrite function as needed for GOSPA calculation, need to include parameter a
def calculate_gospa(est_mat, true_mat, c, p, a, use_empty=True, core_method=None,
                   true_cov_mat=None, est_cov_mat=None):
    """Calculates the OSPA distance between the truth at all timesteps.

    Notes
    -----
    This calculates the Generalized Optimal SubPattern Assignment metric for the
    extracted states and the supplied truth point distributions. The
    calculation is based on
    :cite: `Rahmathullah2017_GeneralizedOptimalSubPatternAssignmentMetric`

    Parameters
    ----------
    est_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`est_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to estimated states.
    true_mat : S x T x N numpy array
        Numpy array of state dimension by number of timesteps by number of objects
        for times/objects which do not exist use a value of np.nan for all state
        dimensions (i.e. if object 1 at timestep 2 does not exist then
        :code:`true_mat[:, 2, 1] = np.nan * np.ones(state_dim)`). This
        corresonds to the true states.
    c : float
        Distance cutoff for considering a point properly assigned. This
        influences how cardinality errors are penalized. For :math:`p = 1`
        it is the penalty given false point estimate.
    p : int
        The power of the distance term. Higher values penalize outliers
        more.
    a : int
        The normalization factor of the distance term. Appropriately penalizes missed
        or false detection of tracks rather than normalizing by the total maximum
        cardinality.
    use_empty : bool, Optional
        Flag indicating if empty values should be set to 0 or nan. The default
        of True is fine for most cases.
    core_method : :class:`.enums.SingleObjectDistance`, Optional
        The main distance measure to use for the localization component.
        The default value of None implies :attr:`.enums.SingleObjectDistance.EUCLIDEAN`.
    true_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`true_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the true states, the object order must be consistent
        with the truth matrix, and is only needed for core methods
        :attr:`.enums.SingleObjectDistance.HELLINGER`. The default value is None.
    est_cov_mat : S x S x T x N numpy array, Optional
        Numpy array of state dimension by state dimension by number of timesteps
        by number of objects for times/objects which do not exist use a value
        of np.nan for all state dimensions (i.e. if object 1 at timestep 2 does
        not exist then
        :code:`est_cov_mat[:, :, 2, 1] = np.nan * np.ones((state_dim, state_dim))`).
        This corresonds to the estimated states, the object order must be consistent
        with the estimated matrix, and is only needed for core methods
        :attr:`.enums.SingleObjectDistance.MAHALANOBIS`. The default value is None.

    Returns
    -------
    ospa : numpy array
        GOSPA values at each timestep.
    localization : numpy array
        Localization component of the GOSPA value at each timestep.
    cardinality : numpy array
        Cardinality component of the GOSPA value at each timestep.
    core_method : :class:`.enums.SingleObjectDistance`
        Method to use as the core distance statistic.
    c : float
        Maximum distance value used.
    p : int
        Power of the distance term used.
    distances : Ne x Nt x T numpy array
        Numpy array of distances, rows are estimated objects columns are truth.
    e_exists : Ne x T numpy array
        Bools indicating if the estimated object exists at that time.
    t_exists : Nt x T numpy array
        Bools indicating if the true object exists at that time.
    """
    # error checking on optional input arguments
    if core_method is None:
        core_method = SingleObjectDistance.EUCLIDEAN

    elif core_method is SingleObjectDistance.MAHALANOBIS and est_cov_mat is None:
        msg = 'Must give estimated covariances to calculate {:s} OSPA. Using {:s} instead'
        warnings.warn(msg.format(core_method, SingleObjectDistance.EUCLIDEAN))
        core_method = SingleObjectDistance.EUCLIDEAN

    elif core_method is SingleObjectDistance.HELLINGER and true_cov_mat is None:
        msg = 'Must save covariances to calculate {:s} OSPA. Using {:s} instead'
        warnings.warn(msg.format(core_method, SingleObjectDistance.EUCLIDEAN))
        core_method = SingleObjectDistance.EUCLIDEAN

    if core_method is SingleObjectDistance.HELLINGER:
        c = np.min([1, c]).item()

    # setup data structuers
    t_exists = np.logical_not(np.isnan(true_mat[0, :, :])).T
    e_exists = np.logical_not(np.isnan(est_mat[0, :, :])).T

    # compute distance for all permutations
    num_timesteps = true_mat.shape[1]
    nt_objs = true_mat.shape[2]
    ne_objs = est_mat.shape[2]
    distances = np.nan * np.ones((ne_objs, nt_objs, num_timesteps))
    comb = np.array(np.meshgrid(np.arange(ne_objs, dtype=int),
                                np.arange(nt_objs, dtype=int))).T.reshape(-1, 2)
    e_inds = comb[:, 0]
    t_inds = comb[:, 1]
    shape = (ne_objs, nt_objs)

    localization = np.nan * np.ones(num_timesteps)
    cardinality = np.nan * np.ones(num_timesteps)

    for tt in range(num_timesteps):
        # use proper core method
        if core_method is SingleObjectDistance.EUCLIDEAN:
            distances[:, :, tt] = np.sqrt(np.sum((true_mat[:, tt, t_inds]
                                                  - est_mat[:, tt, e_inds])**2,
                                                 axis=0)).reshape(shape)

        elif core_method is SingleObjectDistance.MANHATTAN:
            distances[:, :, tt] = np.sum(np.abs(true_mat[:, tt, t_inds]
                                                - est_mat[:, tt, e_inds]),
                                         axis=0).reshape(shape)

        elif core_method is SingleObjectDistance.HELLINGER:
            for row, col in zip(e_inds, t_inds):
                if not (e_exists[row, tt] and t_exists[col, tt]):
                    continue

                distances[row, col, tt] = calculate_hellinger(est_mat[:, tt, row],
                                                              true_mat[:, tt, col],
                                                              est_cov_mat[:, :, tt, row],
                                                              true_cov_mat[:, :, tt, col])

        elif core_method is SingleObjectDistance.MAHALANOBIS:
            for row, col in zip(e_inds, t_inds):
                if not (e_exists[row, tt] and t_exists[col, tt]):
                    continue

                distances[row, col, tt] = calculate_mahalanobis(est_mat[:, tt, row],
                                                                true_mat[:, tt, col],
                                                                est_cov_mat[:, :, tt, row])

        else:
            warnings.warn('Single-object distance {} is not implemented. SKIPPING'.format(core_method))
            core_method = None
            break

        # check for mismatch
        one_exist = np.logical_xor(e_exists[:, [tt]], t_exists[:, [tt]].T)
        empty = np.logical_and(np.logical_not(e_exists[:, [tt]]),
                               np.logical_not(t_exists[:, [tt]]).T)

        distances[one_exist, tt] = c
        if use_empty:
            distances[empty, tt] = 0
        else:
            distances[empty, tt] = np.nan

        distances[:, :, tt] = np.minimum(distances[:, :, tt], c)

        m = np.sum(e_exists[:, tt])
        n = np.sum(t_exists[:, tt])
        if n.astype(int) == 0 and m.astype(int) == 0:
            localization[tt] = 0
            cardinality[tt] = 0
            continue

        if n.astype(int) == 0 or m.astype(int) == 0:
            localization[tt] = 0
            cardinality[tt] = c
            continue

        cont_sub = distances[e_exists[:, tt], :, tt][:, t_exists[:, tt]]**p
        row_ind, col_ind = linear_sum_assignment(cont_sub)
        cost = cont_sub[row_ind, col_ind].sum()

        card_diff = np.abs(n - m)
        inv_p = 1. / p
        c_p = c**p
        localization[tt] = (cost)**inv_p
        cardinality[tt] = ((c_p / a) * card_diff)**inv_p

    ospa = localization + cardinality

    return (ospa, localization, cardinality, core_method, c, p,
            distances, e_exists, t_exists)