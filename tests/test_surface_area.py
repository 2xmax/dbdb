from unittest import TestCase

import numpy as np

import bet
import dbdb


class TestSurface_area(TestCase):

    def test_surface_area(self):
        pressure_limit_min = 0.05
        pressure_limit_max = 0.305
        n2_csa = 0.162e-18  # units TBD

        data = np.genfromtxt("./../data/maximov2019lang/sampleR.tsv", names=True)
        p = data["p_rel"]
        n = data["Q_cm3_per_g_STP"] / (dbdb.V_m * 1e3)
        s_a = bet.surface_area(p, n,
                               csa=n2_csa,
                               pressure_limit_min=pressure_limit_min,
                               pressure_limit_max=pressure_limit_max)
        self.assertAlmostEqual(5.9, s_a, delta=0.1)  # m2/g

        data = np.genfromtxt("../data/maximov2019lang/sample1-ads.tsv", names=True)
        p = data["p_rel"]
        n = data["Q_cm3_per_g_STP"] / (dbdb.V_m * 1e3)
        s_a = bet.surface_area(p, n,
                               csa=n2_csa,
                               pressure_limit_min=pressure_limit_min,
                               pressure_limit_max=pressure_limit_max)
        self.assertAlmostEqual(31.2, s_a, delta=0.1)  # m2/g

        # -- argon --
        # Fig. 2 in Gardner L et al, JPC 2001 https://doi.org/10.1021/jp011745+
        data = np.genfromtxt("../data/arc/BP_fig2_87K.tsv", names=True)
        p = data["p_rel"]
        n = data["Q_cm3_per_g_STP"] / (dbdb.V_m * 1e3)
        arc_csa = 0.138e-18  # m2
        s_a = bet.surface_area(p, n,
                               csa=arc_csa,
                               pressure_limit_min=0.06,
                               pressure_limit_max=0.2)
        self.assertAlmostEqual(35.9, s_a, delta=1)  # m2/g
