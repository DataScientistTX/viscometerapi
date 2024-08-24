import numpy as np
from scipy.optimize import curve_fit

def YPLfunction(y, ty, K, n):
    return ty + K * y ** n

def PLfunction(y, K, n):
    return K * y ** n

def BPfunction(y, PV, YP):
    return YP + PV * y

def r2(residuals, shear_stress, shear_rate):
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((shear_stress - np.mean(shear_stress)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def PL(shear_stress, shear_rate):
    popt, pcov = curve_fit(PLfunction, shear_rate, shear_stress)
    K, m = popt[0], popt[1]
    residuals = shear_stress - PLfunction(shear_rate, popt[0], popt[1])
    r_squared = r2(residuals, shear_stress, shear_rate)
    return K, m, r_squared

def YPL(shear_stress, shear_rate):
    popt, pcov = curve_fit(YPLfunction, shear_rate, shear_stress)
    ty, K, m = popt[0], popt[1], popt[2]
    residuals = shear_stress - YPLfunction(shear_rate, popt[0], popt[1], popt[2])
    r_squared = r2(residuals, shear_stress, shear_rate)

    if popt[0] < 0:
        K, m, r_squared = PL(shear_stress, shear_rate)
        ty = 0
    return ty, K, m, r_squared

def BP(shear_stress, shear_rate):
    PV = (shear_stress[0] - shear_stress[1]) / 511
    YP = 2 * shear_stress[1] - shear_stress[0]
    residuals = shear_stress - BPfunction(shear_rate, PV, YP)
    r_squared = r2(residuals, shear_stress, shear_rate)

    sigma600 = (YP + PV * 600 * 1.7) / (1.066 * 0.4788)
    sigma300 = (YP + PV * 300 * 1.7) / (1.066 * 0.4788)
    return r_squared, PV, YP, sigma600, sigma300