from unittest import TestCase

import numpy as np
import scipy
from CoolProp.CoolProp import PropsSI as props

import dbdb


def mape(a, b):
    return 100 * np.abs((a - b) / b).mean()


class TestIsotherm(TestCase):

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        species = "Argon"
        s_a_black_pearl_87K = 35.9e3  # m2/kg

        P_atm = 101325  # Pa
        T = np.round(props("T", "P", P_atm, "Q", 0, species), 2)
        gamma = props("SURFACE_TENSION", "T", T, "P", P_atm, species)  # N/m
        V_l = 1.0 / props("Dmolar", "T", T, "P", P_atm, species)  # m3/mol
        self.fhh_k = 41.831839393006156
        self.fhh_m = 2.247650098624232
        self.inst = dbdb.Dbdb(T=T, V_l=V_l, gamma=gamma, reference_s_a=s_a_black_pearl_87K)

    def test_arc(self):
        for pore_size in [10, 20, 30, 40]:
            if pore_size < 30:
                pore_sizes = np.arange(1, 56, 2)
            else:
                pore_sizes = np.arange(20, 56, 2)
            data = np.genfromtxt("../data/arc/arc_%dnm.tsv" % pore_size, names=True)
            kernel = self.inst.kernel(data["P"], pore_sizes, self.fhh_k, self.fhh_m)
            n_ads = data["V"] / np.max(data["V"])
            psd_nnls = scipy.optimize.nnls(kernel, n_ads)[0]
            predicted_pore_size = pore_sizes[np.argmax(psd_nnls)]
            self.assertAlmostEqual(pore_size, predicted_pore_size, delta=5)
