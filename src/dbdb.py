#!/usr/bin/env python3

import numpy as np
import scipy.stats
from scipy.integrate import quad
from scipy.optimize import fsolve

P_atm = 101325  # atm pressure, Pa
R_g = 8.314459848  # gas const, J/(mol * K)
V_m = 22.414e-3  # Volume STP, m3/mol
h_0 = 1e-10  # thickness const (to make h dimensionless), m


class Dbdb:
    def __init__(self, T, V_l, gamma, reference_s_a):
        """

        :param T: temperature, K
        :param V_l: molar volume, m^3/mol
        :param gamma: surface tension, N/m
        :param reference_s_a: surface area of the nonporous material, m^2/kg
        """
        self.T = T
        self.V_l = V_l
        self.gamma = gamma
        self.reference_s_a = reference_s_a

    def thickness(self, density, r):
        return r * (1 - ((1 - density) ** (1 / 3.0)))

    def disjoining_pressure(self, h, k, m):
        return (R_g * self.T / self.V_l) * k / ((h / h_0) ** m)

    def relative_pressure(self, h, k, m, r):
        """h, r in meters"""
        mu = -(self.disjoining_pressure(h, k, m) + 2 * self.gamma / (r - h)) * self.V_l
        return np.exp(mu / (R_g * self.T))

    def condensation_point(self, k, m, pore_size):
        r = 0.5 * pore_size

        def dmu_dh_loss(h):
            dmu_dh = m * R_g * self.T * k * (h_0 ** m) * (h ** (-m - 1)) - 2 * self.gamma * self.V_l * ((r - h) ** -2)
            return dmu_dh

        x_0 = self.thickness(0.2, r)  # arbitrary non-zero initial value
        h_sol = fsolve(dmu_dh_loss, x0=x_0)
        p_c = self.relative_pressure(h_sol, k, m, r)
        n_c = 1 - (1 - h_sol / r) ** 3
        return p_c, n_c

    def evaporation_point(self, k, m, pore_size):
        r = 0.5 * pore_size

        scale = 1e10

        def mu_loss(h_scaled):
            h = h_scaled / scale

            # print(f"trial: h = {np.round(h * 1e9, 5)} nm")

            def integrand(h_prime):
                return ((r - h_prime) ** 2) * self.disjoining_pressure(h_prime, k, m)

            # constraints for solver
            if np.isnan(h) or h <= 0 or h >= r:
                return np.inf

            p_rel = self.relative_pressure(h, k, m, r)
            if p_rel <= 0:
                return np.inf

            mu = R_g * self.T * np.log(p_rel)
            broekhoff67_eq17 = (-3 * self.V_l / (r - h)) * (
                    self.gamma + (1.0 / (r - h) ** 2) * quad(integrand, h, r)[0])
            return mu - broekhoff67_eq17

        x_0 = self.thickness(0.2, r) * scale  # arbitrary non-zero initial value
        bounds = (0, r * scale)
        # h_sol = fsolve(mu_loss, x0=x_0 * scale)
        # print(f"h0 = {x_0 * 1e9:.2f} nm, bounds = {bounds[0] * 1e9:.2f} .. {bounds[1] * 1e9:.2f}")
        # h_sol = scipy.optimize.root(mu_loss, x0=x_0).x
        # h_sol = scipy.optimize.differential_evolution(mu_loss, x_0, bounds=bounds).x
        h_sol = scipy.optimize.least_squares(mu_loss, x0=x_0, bounds=bounds).x
        loss = np.abs(mu_loss(h_sol[0]))
        h_sol = h_sol / scale
        if loss > 1e-3 or np.isnan(loss):
            log_msg = f"d = {pore_size * 1e9:.1f} nm, loss = {loss:.4f}, h_sol = {h_sol[0] * 1e9:.3f} nm"
            # print(f"fail: {log_msg}")
            raise Exception(f"loss is too large {log_msg}")

        # else:
        #     print(f"ok: {log_msg}")

        p_e = self.relative_pressure(h_sol, k, m, r)
        n_e = 1 - (1 - h_sol / r) ** 3
        return p_e, n_e

    def isotherm(self, fhh_k, fhh_m, pore_size, raw=False, is_ads_branch=True, add_p=True, density=None):
        """
        returns DbDB isotherm

        :param fhh_k: FHH fitting param k
        :param fhh_m: FHH fitting param m
        :param pore_size: pore diameter in meters
        :param raw: return the raw curve or approximate the sharp increase as n_c -> 1?
        :param is_ads_branch: adsorption or desorption branch?
        :param add_p: add the last point p, n = (1, 1)?
        :param density: density step
        :return: relative pressure and density
        """

        r = 0.5 * pore_size

        if density is None:
            # todo: calculate the monolayer density
            density = np.linspace(0, 1, 1001)[1:]
            if raw:
                # division by zero at r == h (happens for raw curves only)
                density = np.linspace(0, 1, 1001)[1:-1]
                density = np.concatenate((density, [1 - 1e-8]))

        n = np.copy(density)
        h = self.thickness(n, r)

        if not raw:
            if is_ads_branch:
                p_cut, n_cut = self.condensation_point(fhh_k, fhh_m, pore_size)
            else:
                p_cut, n_cut = self.evaporation_point(fhh_k, fhh_m, pore_size)

            p = np.zeros(len(n))
            p[n < n_cut] = self.relative_pressure(h[n < n_cut], fhh_k, fhh_m, r)
            p[n >= n_cut] = p_cut

            if add_p:
                p = np.concatenate((p, [1]))
                n = np.concatenate((n, [1]))
        else:
            p = self.relative_pressure(h, fhh_k, fhh_m, r)

        return p, n

    def kernel(self, rel_pressure_range, pore_sizes_nm, fhh_k, fhh_m, is_ads_branch=True):
        def isotherm_p_interp(pore_size_m):
            n_range = np.linspace(0, 1, 10001)[1:-1]
            # n_range = np.linspace(0, 1, 11)[1:-1]

            if is_ads_branch:
                p_cut, n_cut = self.condensation_point(fhh_k, fhh_m, pore_size_m)
            else:
                p_cut, n_cut = self.evaporation_point(fhh_k, fhh_m, pore_size_m)

            p, n = self.isotherm(fhh_k, fhh_m, pore_size_m, density=n_range, is_ads_branch=is_ads_branch)
            # p, n = add_compressibility(p, n, p_cut)
            n_interp = scipy.interpolate.interp1d(p, n, fill_value="extrapolate")(rel_pressure_range)
            n_interp[np.where(rel_pressure_range > p_cut)] = 1
            return n_interp

        X_columns = ()
        for pore_size in pore_sizes_nm:
            ret_n = isotherm_p_interp(pore_size * 1e-9)
            X_columns = X_columns + (ret_n,)

        X = np.column_stack(X_columns)
        return X

    def single_mode_pore_size(self, X, pore_sizes, p, n_exp):
        """

        :param X: kernel
        :param pore_sizes: any units, the returned value will correspond to the value
        :param p: relative pressure range
        :param n_exp: relative density (will be converted to relative units anyway)
        :return:
        """
        is_sorted = np.all(np.diff(p) >= 0)
        if not is_sorted:
            raise ValueError("the algorithm assumes the input data has to be sorted by pressure")

        n_exp = n_exp / np.max(n_exp)
        p_cut_kernel = np.zeros(len(pore_sizes))
        for psz_idx in range(len(pore_sizes)):
            n_kernel = X[:, psz_idx]
            p_cut_idx = np.argmax(np.diff(n_kernel))
            p_cut_kernel[psz_idx] = p[p_cut_idx]

        # lin interpolate and find where the slope of the isotherm is highest
        n_interp = scipy.interpolate.interp1d(p, n_exp, fill_value="extrapolate")
        p_interp_step = np.linspace(0, 1, 1000)
        dndp_max_idx = np.argmax(np.diff(n_interp(p_interp_step)))
        p_cut_exp = p_interp_step[dndp_max_idx]
        p_cut_psz_idx = np.argmin(np.abs(p_cut_kernel - p_cut_exp))

        return pore_sizes[p_cut_psz_idx]

    # relative pressure for given k, m FHH fitting params
    def relative_pressure_macropore(self, n_ads, k, m):
        h = n_ads * self.V_l / self.reference_s_a
        return np.exp(-k / ((h / h_0) ** m))

    def fhh_fit(self, p_rel, n_ads):
        """
        Frenkel-Halsey-Hill isotherm fit
        :param p_rel: relative pressure
        :param n_ads: amount adsorbed
        :return: k, m fitting params
        """
        h = n_ads * self.V_l / self.reference_s_a
        x = np.log(h / h_0)
        y = np.log(-np.log(p_rel))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        k = np.exp(intercept)  # FHH fitting parameter, dimensionless
        m = -slope  # FHH fitting parameter, dimensionless
        return k, m


def nitrogen(T=77, V_l=3.466e-5, gamma=8.88e-3, reference_s_a=5.9e3):
    return Dbdb(T, V_l=V_l, gamma=gamma, reference_s_a=reference_s_a)
