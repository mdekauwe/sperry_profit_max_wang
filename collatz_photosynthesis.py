#!/usr/bin/env python

"""
Collatz photosynthesis model.


That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (11.01.2020)"
__email__ = "mdekauwe@gmail.com"

import sys
import numpy as np
import os
import math
import constants as cnt

class CollatzC3(object):

    """
    Collatz photosynthesis model.


    Reference
    =========
    * Collatz, G. J., Ball, J. T., Grivet, C., and Berry, J. A. (1991)
      Physiological and environmental regulation of stomatal conductance,
      photosynthesis and transpiration: amodel that includes alaminar boundary
      layer, Agr. Forest Meteorol., 54, 107–136.
    * Clark DB, Mercado LM, Sitch S, Jones CD, Gedney N, Best MJ, Pryor M,
      Rooney GG, Essery RLH, Blyth E, et al. 2011. The Joint UK Land
      Environment Simulator (JULES), Model description – Part 2: Carbon fluxes
      and vegetation. Geoscientific Model Development Discussions 4: 641–688.

    """
    # Note in Clark et al. Ko25=30.0*1E4, using 1E3 to match Eller, check if
    # that is a mistake
    def __init__(self, Oa=21000.0, gamstar25=42.75, Kc25=30.0, Ko25=30.0*1E3,
                 Q10_Kc=2.1, Q10_Ko=1.2, Q10_Vcmax=2.0, Tlower=10.0,
                 Tupper=50.0, gamma25=2600.0, Q10_gamma=0.57, alpha=0.08,
                 omega=0.15, beta1=0.83, beta2=0.93):

        self.gamma25 = gamma25 # coefficents for CO2 compensation point (Pa)
        self.Kc25 = Kc25 # MM coefficents for carboxylation by Rubisco (Pa)
        self.Ko25 = Ko25 # MM coefficents for oxygenation by Rubisco (Pa)
        self.Q10_Ko = Q10_Ko # Q10 value for MM constants for O2
        self.Q10_Kc = Q10_Kc # Q10 value for MM constants for CO2
        self.Q10_Vcmax = Q10_Vcmax # Q10 value for carboxylation of Rubisco
        self.Q10_gamma = Q10_gamma # Q10 value for Rubisco specificity for CO2
                                   # relative to O2
        self.Tlower = Tlower # Lower temperature for carboxylation
        self.Tupper = Tupper # Upper temperature for carboxylation
        self.Oa = Oa # the partial pressure of atmospheric oxygen (Pa)
        self.alpha = alpha # quantum efficiency of
                           # photosynthesis (mol CO2 mol-1 PAR)
        self.omega = omega #  leaf scattering coefficent for PAR (unitless)
        self.beta1 = beta1 # smoothing co-limitation coefficient
        self.beta2 = beta2 # smoothing co-limitation coefficient

    def calc_photosynthesis(self, Ci, Tleaf, PAR, Vcmax25):
        """

        Parameters
        ----------
        Ci : float
            leaf intercellular CO2 partial pressure (Pa)
        Tleaf : float
            leaf temp (deg C)
        PAR : float
            photosynthetically active radiation (mol m-2 s-1)
        Vcmax25 : float
            Maximum rate of rubisco activity 25C (mol m-2 s-1)

        """
        Tk = Tleaf + cnt.DEG_2_KELVIN

        # CO2 compensation point in the absence of mitochondrial resp (Pa)
        gamma = self.calc_CO2_compensation_point(Tleaf)

        # calculate temp depend of Michaelis Menten constants for CO2, O2
        Km = self.calc_michaelis_menten_constants(Tleaf)

        # Max rate of rubisco activity (mol m-2 s-1)
        Vcmax = self.correct_vcmax_for_temperature(Vcmax25, Tleaf)

        # Leaf day respiration (mol m-2 s-1)
        Rd = Vcmax * 0.01

        # Leaf-level photosynthesis: Rubisco-limited rate (Pa)
        Ac = Vcmax * ((Ci - gamma) / (Ci + Km))

        # Leaf-level photosynthesis: Light-limited rate (Pa)
        Al = self.alpha * (1.0 - self.omega) * PAR \
                * ((Ci - gamma) / (Ci + 2.0 * gamma))

        # Leaf-level photosynthesis: rate of transport of photosynthetic
        # products
        Ae = 0.5 * Vcmax

        # The rate of gross photosynthesis (W) is calculated as the smoothed
        # minimum of three potentially-limiting rates
        A = self.beta1
        B = -(Ac + Al)
        C = Ac * Al
        A_gross1 = self.quadratic(a=A, b=B, c=C, large=False)

        A = self.beta2
        B = -(A_gross1 + Ae)
        C = A_gross1 * Ae
        A_gross2 = self.quadratic(a=A, b=B, c=C, large=False)

        # Rate of gross photosynthesis (mol CO2 m-2 s-1)
        Ag = A_gross2

        # Rate of net photosynthesis (mol CO2 m-2 s-1)
        An = Ag - Rd

        return An

    def calc_photosynthesis_given_gc(self, Cs, Tleaf, PAR, Vcmax25, gc, press):
        """

        Parameters
        ----------
        Cs : float
            leaf CO2 partial pressure (Pa)
        Tleaf : float
            leaf temp (deg C)
        PAR : float
            photosynthetically active radiation (mol m-2 s-1)
        Vcmax25 : float
            Maximum rate of rubisco activity 25C (mol m-2 s-1)
        gc : float
            stomatal conductance to CO2
        press: float
            atmospheric pressure (Pa)
        """
        Tk = Tleaf + cnt.DEG_2_KELVIN

        # CO2 compensation point in the absence of mitochondrial resp (Pa)
        gamma = self.calc_CO2_compensation_point(Tleaf)

        # calculate temp depend of Michaelis Menten constants for CO2, O2
        Km, Ko, Kc = self.calc_michaelis_menten_constants(Tleaf, ret_cnts=True)

        # Max rate of rubisco activity (mol m-2 s-1)
        Vcmax = self.correct_vcmax_for_temperature(Vcmax25, Tleaf)

        # Leaf day respiration (mol m-2 s-1)
        Rd = Vcmax * 0.01

        # Leaf-level photosynthesis: Rubisco-limited rate (Pa)
        a = Vcmax
        b = Kc * (1.0 + self.Oa / Ko)
        c = (Rd - a) - (gc / press) * (Cs + b)
        d = (gc / press) * (a * Cs - a * gamma - Rd * Cs - b * Rd)
        Ac = (-(c / 2.0) - math.sqrt(((c / 2.0)**2) - d)) + Rd

        # Leaf-level photosynthesis: Light-limited rate (Pa)
        a = self.alpha * (1.0 - self.omega) * PAR
        b = 2.0 * gamma
        c = (Rd - a) - (gc / press) * (Cs + b)
        #d = (gc/Pa)*(a * Ca - a * photocomp - Rd * Ca - b*Rd)
        d = (gc / press) * (a * Cs - a * gamma - Rd * Cs - b * Rd)
        Al = (-(c / 2.0) - math.sqrt(((c / 2.0)**2.0) - d)) + Rd

        # Leaf-level photosynthesis: rate of transport of photosynthetic
        # products
        Ae = 0.5 * Vcmax

        # Co-limitation

        # The rate of gross photosynthesis (W) is calculated as the smoothed
        # minimum of three potentially-limiting rates
        A = self.beta1
        B = -(Ac + Al)
        C = Ac * Al
        A_gross1 = self.quadratic(a=A, b=B, c=C, large=False)

        A = self.beta2
        B = -(A_gross1 + Ae)
        C = A_gross1 * Ae
        A_gross2 = self.quadratic(a=A, b=B, c=C, large=False)

        # Rate of gross photosynthesis (mol CO2 m-2 s-1)
        Ag = A_gross2

        # Rate of net photosynthesis (mol CO2 m-2 s-1)
        An = Ag - Rd

        return An

    def calc_ci_at_colimitation_point(self, Ci, Tleaf, PAR, Vcmax25):
        """

        Parameters
        ----------
        Ci : float
            leaf intercellular CO2 partial pressure (Pa)
        Tleaf : float
            leaf temp (deg C)
        PAR : float
            photosynthetically active radiation (mol m-2 s-1)
        Vcmax25 : float
            Maximum rate of rubisco activity 25C (mol m-2 s-1)

        """
        Tk = Tleaf + cnt.DEG_2_KELVIN

        # CO2 compensation point in the absence of mitochondrial resp (Pa)
        gamma = self.calc_CO2_compensation_point(Tleaf)

        # calculate temp depend of Michaelis Menten constants for CO2, O2
        Km, Ko, Kc = self.calc_michaelis_menten_constants(Tleaf, ret_cnts=True)

        # Max rate of rubisco activity (mol m-2 s-1)
        Vcmax = self.correct_vcmax_for_temperature(Vcmax25, Tleaf)

        # Leaf day respiration (mol m-2 s-1)
        Rd = Vcmax * 0.01

        # Leaf-level photosynthesis: Light-limited rate (Pa)
        Al = self.alpha * (1.0 - self.omega) * PAR

        # Leaf-level photosynthesis: rate of transport of photosynthetic
        # products
        Ae = 0.5 * Vcmax

        # Co-limitated A
        A = self.beta2
        B = -(Al + Ae)
        C = Al * Ae
        A_colimit = self.quadratic(a=A, b=B, c=C, large=False)

        # Ci at the colimitation point
        a_bnd = -Vcmax * gamma
        b_bnd = Vcmax
        d_bnd = Kc * (1.0 + self.Oa / Ko)
        e_bnd = 1.0
        Ci_col = (a_bnd - d_bnd * A_colimit) / (e_bnd * A_colimit - b_bnd)

        return Ci_col

    def calc_michaelis_menten_constants(self, Tleaf, ret_cnts=False):
        """
        Michaelis-Menten constant for O2/CO2, Arrhenius temp dependancy

        Parameters:
        ----------
        Tleaf : float
            leaf temperature [deg K]

        Returns:
        ----------
        Km : float
            Michaelis-Menten constant
        """
        # Michaelis Menten constants for CO2 (Pa)
        Kc = self.Q10_func(self.Kc25, self.Q10_Kc, Tleaf)

        # Michaelis Menten constants for O2 (Pa)
        Ko = self.Q10_func(self.Ko25, self.Q10_Ko, Tleaf)

        Km = Kc * (1.0 + self.Oa / Ko)

        if ret_cnts:
            return Km, Ko, Kc
        else:
            return Km

    def calc_CO2_compensation_point(self, Tleaf):
        """
        Photorespiration compensation point (Pa)
        """

        # Rubisco specificity for CO2 relative to O2
        tau = self.Q10_func(self.gamma25, self.Q10_gamma, Tleaf)
        gamma = self.Oa / (2.0 * tau)

        return gamma

    def Q10_func(self, k25, Q10, Tleaf):
        """
        Q10 function to calculate parameter change with temperature
        """

        return k25 * (Q10**((Tleaf - 25.0) / 10.0))

    def correct_vcmax_for_temperature(self, Vcmax25, Tleaf):
        """
        Correct Vcmax based on defined by PFT-specific upper and lower
        temperature params, see Clark et al. (mol CO2 m-2 s-1)
        """
        num = self.Q10_func(Vcmax25, self.Q10_Vcmax, Tleaf)
        den = (1.0 + math.exp(0.3 * (Tleaf - self.Tupper))) * \
                (1.0 + math.exp(0.3 * (self.Tlower - Tleaf)))

        return num / den

    def quadratic(self, a=None, b=None, c=None, large=False):
        """ minimilist quadratic solution as root for J solution should always
        be positive, so I have excluded other quadratic solution steps. I am
        only returning the smallest of the two roots

        Parameters:
        ----------
        a : float
            co-efficient
        b : float
            co-efficient
        c : float
            co-efficient

        Returns:
        -------
        val : float
            positive root
        """
        d = b**2.0 - 4.0 * a * c # discriminant
        if d < 0.0:
            raise ValueError('imaginary root found')
        #root1 = np.where(d>0.0, (-b - np.sqrt(d)) / (2.0 * a), d)
        #root2 = np.where(d>0.0, (-b + np.sqrt(d)) / (2.0 * a), d)

        if large:
            if math.isclose(a, 0.0) and b > 0.0:
                root = -c / b
            elif math.isclose(a, 0.0) and math.isclose(b, 0.0):
                root = 0.0
                if c != 0.0:
                    raise ValueError('Cant solve quadratic')
            else:
                root = (-b + np.sqrt(d)) / (2.0 * a)
        else:
            if math.isclose(a, 0.0) and b > 0.0:
                root = -c / b
            elif math.isclose(a, 0.0) and math.isclose(b, 0.0):
                root == 0.0
                if c != 0.0:
                    raise ValueError('Cant solve quadratic')
            else:
                root = (-b - np.sqrt(d)) / (2.0 * a)

        return root

if __name__ == "__main__":

    Vcmax25 = 0.0001 # Maximum rate of rubisco activity 25C (mol m-2 s-1)
    Tleaf = 35.0     # Leaf temp (deg C)
    Ci = 40. * 0.7  # leaf interceullular partial pressure (Pa)
    PAR = 0.002      # photosynthetically active radiation (mol m-2 s-1)

    C = CollatzC3()
    An = C.calc_photosynthesis(Ci, Tleaf, PAR, Vcmax25)

    print(An)
