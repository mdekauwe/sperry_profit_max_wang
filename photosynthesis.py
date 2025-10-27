# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:57:45 2018

@author: Yujie
"""

from pylab import array,exp,sqrt

# calculate j from light
def GetPhotosyntheticJ(jmax, light):
    a = 0.9
    b = -0.3*light - jmax
    c = 0.3*light*jmax
    j = ( -b - sqrt(b*b-4*a*c) ) / a * 0.5
    return j

# calculate jmax from temperature
def GetPhotosyntheticJmax(jmax25, tem):
    ha=50300.0
    hd=152044.0
    sv=495.0
    t0=298.15
    r=8.315
    c = 1.0 + exp((sv*t0 -hd)/(r*t0))
    t1 = tem + 273.15
    factor = c * exp(ha/r/t0*(1.0-t0/t1)) / (1.0 + exp((sv*t1-hd)/(r*t1)))
    jmax = jmax25 * factor
    return jmax

# calculate vcmax from temperature
def GetPhotosyntheticVcmax(vcmax25, tem):
    ha=73637.0
    hd=149252.0
    sv=486.0
    t0=298.15
    r=8.315
    c = 1.0 + exp((sv*t0 -hd)/(r*t0))
    t1 = tem + 273.15
    factor = c * exp(ha/r/t0*(1.0-t0/t1)) / (1.0 + exp((sv*t1-hd)/(r*t1)))
    vcmax = vcmax25 * factor
    return vcmax

def get_a(v25,j25,gamma,ci,tem,par):
    adjust = 0.98
    r_day  = calc_rday(v25, tem)
    vmax = GetPhotosyntheticVcmax(v25,tem)
    jmax = GetPhotosyntheticJmax(j25,tem)
    j = GetPhotosyntheticJ(jmax,par)
    kc = 41.01637 * 2.1**(0.1*(tem-25.0))
    ko = 28201.92 * 1.2**(0.1*(tem-25.0))
    km = kc * (1.0+21000.0/ko)
    aj = j * (ci-gamma) / (4.0*(ci+2*gamma))
    ac = vmax * (ci-gamma) / (ci+km)
    af = (aj + ac - sqrt((aj+ac)**2.0 - 4*adjust*aj*ac) ) / adjust * 0.5
    af = af - r_day
    return af

def calc_rday(v25, tem):
    return v25 * 0.01 * 2.0**(0.1*(tem-25.0)) / (1.0+exp(1.3*(tem-55.0)))

# get ci and Anet from gc and ca
def get_a_ci(v25,j25,gamma,gc,ca,tem,par):
    tar_ci  = 0.0
    tar_a  = 0.0
    max_ci  = ca
    min_ci  = gamma
    eps = 1e-12
    
    while True:
        tar_ci = 0.5 * (max_ci + min_ci)
        af = get_a(v25, j25, gamma, tar_ci, tem, par)

        tmp_g = af / (ca - tar_ci) # Pa

        if abs(tmp_g - gc)/(gc + eps) < 1E-12:
            tar_a = af
            break
        elif (tmp_g < gc):
            min_ci = tar_ci
        else:
            max_ci = tar_ci

        if (abs(max_ci - min_ci) < 1E-12):
            tar_a = af
            break

    return tar_ci, tar_a
