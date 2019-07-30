import numpy as np
import scipy.constants
import scipy.stats


def surface_area(p_rel, n, pressure_limit_min=0.05, pressure_limit_max=0.30, *, csa):
    """
    :param p_rel:
    :param n: density, mmol/g
    :param pressure_limit_min: pressure limit
    :param pressure_limit_max: pressure limit
    :param csa: cross-sectional area, m2
    :return: BET surface area, m2/g
    """

    p_filter = np.logical_and(pressure_limit_min <= p_rel, p_rel <= pressure_limit_max)
    p_rel = p_rel[p_filter]
    n = n[p_filter] * 1e-3  # mmol/g -> mol/g
    y = 1 / (n * (1 / p_rel - 1))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(p_rel, y)
    s_a = scipy.constants.N_A * csa / (slope + intercept)
    return s_a
