from unittest import TestCase

import numpy as np

import bet
import dbdb


class TestKernel(TestCase):

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)

    def test_range(self):
        # ensures there is no exception caused by scaling to various pore sizes
        data = np.genfromtxt("../data/maximov2019lang/sampleR.tsv", names=True)
        pressure = data["p_rel"]
        density = data["Q_cm3_per_g_STP"] / (dbdb.V_m * 1e3)
        n2_csa = 0.162e-18  # m2
        s_a = bet.surface_area(pressure, density, csa=n2_csa) * 1e3
        inst = dbdb.nitrogen(reference_s_a=s_a)
        fhh_k, fhh_m = inst.fhh_fit(pressure, density)

        data = np.genfromtxt("../data/maximov2019lang/sample2-des.tsv", names=True)
        pressure_exp = data["p_rel"]
        pore_sizes = np.array([1, 2, 5, 10, 20, 50, 100, 125, 150, 200, 500, 1000, 10000])
        kernel = inst.kernel(pressure_exp, pore_sizes, fhh_k, fhh_m, is_ads_branch=False)
        self.assertEqual([False], np.unique(np.isnan(kernel)))
