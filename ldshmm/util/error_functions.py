import numpy as np
import pyemma
import math

def matrix_err(A, B):
    """
    Matrix error calculation

    :param A: ndarray - transition matrix
    :return: float error
    """

    return np.linalg.norm(A-B.trans)


def timescale_mean_rel_err(A,B):
    """
    Timescale mean error calculation

    :param A: ndarray - transition matrix
    :return: float error
    """

    import numpy as np
    msm_estimated = pyemma.msm.MSM(A)
    ts_actual = msm_estimated.timescales()
    ts_estimated = B.timescales()
    ts_rel_err = [(math.fabs(ts_actual[i] - ts_estimated[i])/ts_actual[i]) for i in range(len(ts_estimated))]
    ts_rel_err_mean = np.mean(ts_rel_err)
    return ts_rel_err_mean



def stat_dist_vec_err(A,B):
    """
    Stationary distribution error calculation

    :param A: ndarray - transition matrix
    :return: float error
    """

    msm_estimated = pyemma.msm.MSM(A)
    pi_actual = msm_estimated._pi
    pi_estimated = B._pi
    pi_vec_err = np.linalg.norm(pi_actual - pi_estimated)
    return pi_vec_err