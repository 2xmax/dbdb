from unittest import TestCase

import numpy as np
from CoolProp.CoolProp import PropsSI as props

import dbdb


class TestFhh_fit(TestCase):
    def test_fhh_fit_opal(self):
        """
        Given macroporous isotherm, estimate the Frenkel−Halsey−Hill fitting params k and m
        :return:
        """
        species = "Nitrogen"
        T = np.round(props("T", "P", dbdb.P_atm, "Q", 0, species), 2)

        inst = dbdb.Dbdb(T=T,
                         V_l=3.466e-5,
                         gamma=props("SURFACE_TENSION", "T", T, "Q", 0, species),  # N/m,
                         reference_s_a=5.9e3)

        exp_data = np.genfromtxt("./../data/maximov2019lang/sampleR.tsv", names=True)
        p_rel = exp_data["p_rel"]
        n_ads = exp_data["Q_cm3_per_g_STP"] / (dbdb.V_m * 1e3)
        k, m = inst.fhh_fit(p_rel, n_ads)
        self.assertAlmostEqual(47.22, k, delta=0.1)
        self.assertAlmostEqual(2.53, m, delta=0.01)

    def test_fhh_fit_arc(self):
        s_a_black_pearl_87K = 35.9e3  # m2/kg
        species = "Argon"
        T = np.round(props("T", "P", dbdb.P_atm, "Q", 0, species), 2)

        inst = dbdb.Dbdb(T=T,
                         V_l=1.0 / props("Dmolar", "T", T, "P", dbdb.P_atm, species),  # m3/mol
                         gamma=props("SURFACE_TENSION", "T", T, "Q", 0, species),  # N/m,
                         reference_s_a=s_a_black_pearl_87K)

        # Fig. 2 in Gardner L et al, JPC 2001 https://doi.org/10.1021/jp011745+
        exp_data = np.genfromtxt("./../data/arc/BP_fig2_87K.tsv", names=True)
        p_rel = exp_data["p_rel"]
        n_ads = exp_data["Q_cm3_per_g_STP"] / (dbdb.V_m * 1e3)
        k, m = inst.fhh_fit(p_rel, n_ads)
        self.assertAlmostEqual(41.831839393006156, k, delta=1e-3)
        self.assertAlmostEqual(2.247650098624232, m, delta=1e-5)
