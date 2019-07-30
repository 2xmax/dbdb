import statistics
from unittest import TestCase

import numpy as np

from src import dbdb


def mape(a, b):
    return 100 * np.abs((a - b) / b).mean()


class TestIsotherm(TestCase):

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)
        self.fhh_k = 47.22
        self.fhh_m = 2.53
        self.inst = dbdb.Dbdb(T=77, V_l=3.466e-5, gamma=8.88e-3, reference_s_a=5.9e3)

    def test_isotherm_ads_condensation_point(self):
        pore_diameter = 35e-9 * 2  # m
        pressure, density = self.inst.isotherm(self.fhh_k, self.fhh_m, pore_diameter)
        p_mode = statistics.mode(pressure)
        paper_condensation_pressure = 0.966331280489  # relative
        self.assertAlmostEqual(paper_condensation_pressure, p_mode, delta=0.01)

    def test_isotherm_des_evaporation_point(self):
        pore_diameter = 27e-9 * 2  # m
        pressure, density = self.inst.isotherm(self.fhh_k, self.fhh_m, pore_diameter, is_ads_branch=False)
        p_mode = statistics.mode(pressure)
        paper_evaporation_pressure = 0.941300930534  # relative
        self.assertAlmostEqual(paper_evaporation_pressure, p_mode, delta=0.01)

    def test_single_mode_pore_size(self):
        test_cases = [
            dict(sample_id=1, ads=True, pore_size=2 * 35),
            dict(sample_id=1, ads=False, pore_size=2 * 27),
            dict(sample_id=2, ads=True, pore_size=2 * 29),
            dict(sample_id=2, ads=False, pore_size=2 * 23),
        ]
        for test_case in test_cases:
            branch = "ads" if test_case["ads"] else "des"
            data = np.genfromtxt(f"../data/sample{test_case['sample_id']}-{branch}.tsv", names=True)
            data.sort(order="p_rel")
            pressure = data["p_rel"]
            density = data["Q_cm3_per_g_STP"]
            pore_sizes = np.arange(2, 100 + 2, 2)  # nm
            kernel = self.inst.kernel(pressure, pore_sizes_nm=pore_sizes, fhh_k=self.fhh_k, fhh_m=self.fhh_m)
            pore_sz_prediction = self.inst.single_mode_pore_size(kernel, pore_sizes, pressure, density)
            self.assertAlmostEqual(test_case["pore_size"], pore_sz_prediction, delta=8)

    def test_condensation_pressure(self):
        pore_diameter = 35e-9 * 2  # m
        p_c, n_c = self.inst.condensation_point(self.fhh_k, self.fhh_m, pore_diameter)
        paper_p_c = 0.966331280489
        self.assertAlmostEqual(paper_p_c, p_c, delta=0.01)

    def test_thickness(self):
        # Given relative density and radius, estimate thickness
        self.assertAlmostEqual(0, self.inst.thickness(density=0, r=35e-9))
        self.assertAlmostEqual(35e-9, self.inst.thickness(density=0, r=35e-9))

    def test_pointwise(self):
        config = [
            {"sample_id": 1, "des_d_nm": 27 * 2, "ads_d_nm": 35 * 2},
            {"sample_id": 2, "des_d_nm": 23 * 2, "ads_d_nm": 29 * 2}
        ]
        for c in config:
            for is_ads_branch in [True, False]:
                branch = "ads" if is_ads_branch else "des"
                data = np.genfromtxt(f"../data/galukhin2019lang/DBdB-sphere-{branch}-sample{c['sample_id']}.csv", names=True,
                                     delimiter=",")
                data.sort(order="pp0")
                data_p = data["pp0"]
                data_n = data[f"relative_{branch}orption"]

                pore_size = c[f"{branch}_d_nm"] * 1e-9
                p, n = self.inst.isotherm(self.fhh_k, self.fhh_m, pore_size, density=data_n,
                                          is_ads_branch=is_ads_branch)
                self.assertAlmostEqual(0, mape(data_n, n[:-1]), delta=1e-8)

                # condensation and evaporation pt check
                p_c_data = statistics.mode(data_p)
                p_c_sol = statistics.mode(p)
                self.assertAlmostEqual(p_c_data, p_c_sol, delta=2e-3)

                # relative value verification, check if mean absolute percentage error is greater 0.25%
                p_c = min(p_c_sol, p_c_data)
                p_head_data = data_p[data_p < p_c]
                p_head_sol = p[p < p_c]
                common_len = min(len(p_head_sol), len(p_head_data))
                p_head_data = p_head_data[:common_len]
                p_head_sol = p_head_sol[:common_len]
                excluded = c["sample_id"] == 2 and not is_ads_branch
                if not excluded:
                    self.assertAlmostEqual(0, mape(p_head_data, p_head_sol), delta=0.25)
                else:
                    self.assertAlmostEqual(0, mape(p_head_data, p_head_sol), delta=10)
                    self.assertAlmostEqual(0, mape(p_head_data[1:], p_head_sol[1:]), delta=5)
