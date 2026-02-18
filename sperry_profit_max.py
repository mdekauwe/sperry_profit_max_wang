#!/usr/bin/env python
"""
Wrapped Sperry optimisation code from Wang et al. with time varying met data
and simple soil water bucket. The code is as in Wang et al., with added units.

Reference:
==========
* Wang, Y., Sperry, J.S., Anderegg, W.R., Venturas, M.D. and Trugman, A.T.
  (2020), A theoretical and empirical assessment of stomatal optimization
  modeling. New Phytol. Accepted Author Manuscript. doi:10.1111/nph.16572

"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (13.04.2020)"
__email__ = "mdekauwe@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
import constants as cons
from generate_met_data import generate_met_data
from photosynthesis import get_a_ci
import pandas as pd
import sys

class ProfitMax(object):

    def __init__(self, sw0=0.5, psi_e=-0.8, theta_sat=0.5, b=6.,
                 soil_depth=1.0, ground_area=1.0, met_timestep=30.,
                 vcmax=61.74, jmax=111.13, laba=1000.0):

        self.k_max = 10.0   # whole tree conductance
        self.b_plant = 2.0  # shape param
        self.c_plant = 5.0  # shape param

        # tree level
        self.laba = laba   # leaf area per basal area, m2 m-2
        self.vcmax = vcmax
        self.jmax = jmax

        self.soil_depth = soil_depth # depth of soil bucket, m
        self.ground_area = ground_area # m
        self.soil_volume = self.ground_area * self.soil_depth # m3
        self.sw0 = sw0 # initial soil volumetric water content (m3 m-3)
        self.psi_e = psi_e # air entry point water potential (MPa)
        self.theta_sat = theta_sat # soil water capacity at saturation (m3 m-3)
        self.b = b # empirical coefficient related to the clay content of the
                   # soil (Cosby et al. 1984).
        self.met_timestep = met_timestep
        self.timestep_sec = 60. * self.met_timestep

    def run_simulation(self, met=None):

        (n, out) = self.initialise_model(met)

        for i in range(1, n):

            (out.opt_a[i],
             out.opt_g[i],
             out.opt_e[i],
             out.opt_p[i]) = self.sperry_optimisation(out.psi_soil[i-1],
                                                      met.vpd[i], met.Ca[i],
                                                      met.tair[i], met.par[i],
                                                      met.press[i])

            out.sw[i] = self.update_sw_bucket(met.precip[i], out.opt_e[i],
                                              out.sw[i-1])


            # Update soil water potential
            out.psi_soil[i] = self.calc_swp(out.sw[i])

        return (out)

    def initialise_model(self, met):
        """
        Set everything up: set initial values, build an output dataframe to
        save things

        Parameters:
        -----------
        met : object
            met forcing variables: day; Ca; par; precip; press; tair; vpd

        Returns:
        -------
        n : int
            number of timesteps in the met file
        out : object
            output dataframe to store things as we go along

        """
        n = len(met)

        out = self.setup_out_df(met)
        out.sw[0] = self.sw0
        out.psi_soil[0] = self.calc_swp(self.sw0)
        out.hod[0] = 0
        out.doy[0] = 0
        out.year[0] = met.year.iloc[0]

        return n, out

    def setup_out_df(self, met):
        """
        Create and output dataframe to save things

        Parameters:
        -----------
        met : object
            met forcing variables: day; Ca; par; precip; press; tair; vpd

        Returns:
        -------
        out : object
            output dataframe to store things as we go along.
        """
        dummy = np.ones(len(met)) * np.nan
        out = pd.DataFrame({'year':dummy,
                            'doy':dummy,
                            'hod':dummy,
                            'opt_a':dummy,
                            'opt_g':dummy,
                            'opt_e':dummy,
                            'opt_p':dummy,
                            'psi_soil':dummy,
                            'sw':dummy})

        return out

    def sperry_optimisation(self, psi_soil, vpd, ca, tair, par, press):
        """
        Optimise A, g assuming a single whole-plant vulnerability. Note e to gs
        assumes perfect coupling and no energy balance.

        Parameters:
        -----------
        psi_soil : float
            soil water potential, MPa
        vpd : float
            vapour pressure deficit, kPa
        tair : float
            air temperature, deg C
        par : float
            photosynthetically active radiation, umol m-2 s-1
        press : float
            air pressure, kPa

        Returns:
        -----------
        opt_a : float
            optimised A, umol m-2 s-1
        opt_gw : float
            optimised gw, mol H2O m-2 s-1
        opt_e : float
            optimised E, mmol H2O m-2 s-1
        opt_p : float
            optimised p_leaf (really total plant), MPa

        """
        # press doesn't vary, so just use first value to convert mol to Pa
        gc_conv = 1.0 / met.press[0] * cons.KPA_2_PA

        e_crit = self.get_e_crit(psi_soil) # kg H2O 30 min-1 m-2 (basal area)
        de = 1.0

        all_e = np.zeros(0)
        all_k = np.zeros(0)
        all_a = np.zeros(0)
        all_p = np.zeros(0)
        all_g = np.zeros(0)

        for i in range(101):

            # Vary e from 0 to e_crit (0.01 is just partioning step)
            e = i * 0.01 * e_crit
            p = self.get_p_leaf(e, psi_soil)

            # Convert e (kg m-2 30min-1) basal area to mol H2O m-2 s-1
            emol = e * (cons.KG_TO_G * cons.G_WATER_2_MOL_WATER /
                        cons.SEC_2_HLFHR / self.laba)

            # assume perfect coupling
            gh = emol / vpd * press # mol H20 m-2 s-1
            gc = gh * cons.GSW_2_GSC
            g = gc * gc_conv # convert to Pa

            c,a = get_a_ci(self.vcmax, self.jmax, 2.5, g, ca, tair, par)
            e_de = e + de
            p_de = self.get_p_leaf(e_de, psi_soil)
            k = de / (p_de - p)

            all_k = np.append(all_k, k)
            all_a = np.append(all_a, a)
            all_p = np.append(all_p, p)
            all_e = np.append(all_e, emol * cons.mol_2_mmol)
            all_g = np.append(all_g, gc * cons.GSC_2_GSW)

        # Locate maximum profit
        gain = all_a / np.max(all_a)
        risk = 1.0 - all_k / np.max(all_k)
        profit = gain - risk
        idx = np.argmax(profit)
        opt_a = all_a[idx]
        opt_gw = all_g[idx]
        opt_e = all_e[idx]
        opt_p = all_p[idx]

        return opt_a, opt_gw, opt_e, opt_p

    def get_e_crit(self, psi_soil):
        """
        Calculate the maximal E beyond which the tree desiccates due to
        hydraulic failure (e_crit)

        Parameters:
        -----------
        psi_soil : float
            soil water potential, MPa

        Returns:
        -----------
        e_crit : float
            kg H2O 30 min-1 m-2 (basal area)

        """

        # P at Ecrit, beyond which tree desiccates
        p_crit = self.b_plant * np.log(1000.0) ** (1.0 / self.c_plant) # MPa
        e_min = 0.0
        e_max = 100.0
        e_crit = 50.0

        while True:
            p = self.get_p_leaf(e_max, psi_soil) # MPa
            if p < p_crit:
                e_max *= 2.0
            else:
                break

        while True:
            e = 0.5 * (e_max + e_min)
            p = self.get_p_leaf(e, psi_soil)
            if abs(p - p_crit) < 1E-3 or (e_max - e_min) < 1E-3:
                e_crit = e
                break
            if p > p_crit:
                e_max = e
            else:
                e_min = e

        # kg H2O 30 min-1 m-2 (basal area)
        return e_crit

    def get_p_leaf(self, transpiration, psi_soil):
        """
        Integrate vulnerability curve across soil–plant system. This is a
        simplification, as opposed to splitting between root-zone, stem & leaf.

        Parameters:
        -----------
        transpiration : float
            kg H2O 30 min-1 m-2 (basal area)
        psi_soil : float
            soil water potential, MPa

        Returns:
        -----------
        p_leaf : float
            integrated vulnerability curve, MPa

        """
        dp = 0.0
        p = psi_soil # MPa
        N = 20 # P range
        for i in range(N): # iterate through the P range

            # Vulnerability to cavitation
            weibull = np.exp(-1.0 * (p / self.b_plant)**self.c_plant)

            # Hydraulic conductance of the element, including vulnerability to
            # cavitation
            k = max(1E-12, self.k_max * weibull * float(N))
            dp += transpiration / k # should have a gravity, height addition
            p = psi_soil + dp

        p_leaf = p

        # MPa
        return p_leaf

    def calc_swp(self, sw):
        """
        Calculate the soil water potential (MPa). The params The parameters b
        and psi_e are estimated from a typical soil moisture release function.

        Parameters:
        -----------
        sw : object
            volumetric soil water content, m3 m-3

        Returns:
        -----------
        psi_soil : float
            soil water potential, MPa

        References:
        -----------
        * Duursma et al. (2008) Tree Physiology 28, 265276, eqn 10
        """
        return self.psi_e * (sw / self.theta_sat)**-self.b # MPa

    def update_sw_bucket(self, precip, water_loss, sw_prev):
        """
        Update the simple bucket soil water balance

        Parameters:
        -----------
        precip : float
            precipitation (kg m-2 s-1)
        water_loss : float
            flux of water out of the soil (transpiration (kg m-2 timestep-1))
        sw_prev : float
            volumetric soil water from the previous timestep (m3 m-3)
        soil_volume : float
            volume soil water bucket (m3)

        Returns:
        -------
        sw : float
            new volumetric soil water (m3 m-3)
        """
        """
        loss = water_loss * cons.MMOL_2_MOL * cons.MOL_WATER_2_G_WATER * \
                cons.G_TO_KG * self.timestep_sec
        delta_sw = (precip * self.timestep_sec) - loss

        sw = min(self.theta_sat, \
                 sw_prev + delta_sw / (self.soil_volume * cons.M_2_MM))
        sw = max(sw, 0.0)

        return sw
        """
        loss_kg = water_loss * 1e-3 * 0.018 * self.timestep_sec
        precip_kg = precip * self.timestep_sec
        delta_sw = (precip_kg - loss_kg) / (self.soil_depth * 1000)
        sw = sw_prev + delta_sw
        return np.clip(sw, 0.0, self.theta_sat)


if __name__ == "__main__":

    time_step = 30
    lat = -35.76
    lon = 148.0


    met = generate_met_data(Tmin=10, Tmax=30.0, RH=30, ndays=1,
                            lat=lat, lon=lon, time_step=time_step)

    

    # Convert to Pa
    met.Ca *= cons.umol_to_mol * met.press * cons.KPA_2_PA

    S = ProfitMax()
    out = S.run_simulation(met)

    plt.plot(out.opt_a[0:48])
    plt.show()
