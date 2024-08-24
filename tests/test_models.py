import numpy as np
from app.models import YPLfunction, PLfunction, BPfunction, r2, PL, YPL, BP

def test_YPLfunction():
    assert round(YPLfunction(10, 2, 3, 0.5), 2) == 7.39

def test_PLfunction():
    assert round(PLfunction(10, 2, 0.5), 2) == 6.32

def test_BPfunction():
    assert round(BPfunction(10, 2, 3), 2) == 23.00

def test_r2():
    residuals = np.array([1, 2, 3])
    shear_stress = np.array([10, 20, 30])
    shear_rate = np.array([1, 2, 3])
    assert round(r2(residuals, shear_stress, shear_rate), 2) == 0.94

def test_PL():
    shear_stress = np.array([10, 20, 30])
    shear_rate = np.array([1, 2, 3])
    K, n, r_squared = PL(shear_stress, shear_rate)
    assert round(K, 2) == 10.00
    assert round(n, 2) == 1.00
    assert round(r_squared, 2) == 1.00

def test_YPL():
    shear_stress = np.array([10, 20, 30])
    shear_rate = np.array([1, 2, 3])
    ty, K, n, r_squared = YPL(shear_stress, shear_rate)
    assert round(ty, 2) == 0.00
    assert round(K, 2) == 10.00
    assert round(n, 2) == 1.00
    assert round(r_squared, 2) == 1.00

def test_BP():
    shear_stress = np.array([20, 40, 60])
    shear_rate = np.array([300, 600, 900])
    r_squared, PV, YP, sigma600, sigma300 = BP(shear_stress, shear_rate)
    assert round(r_squared, 2) == 1.00
    assert round(PV, 2) == 39.14
    assert round(YP, 2) == 40.00
    assert round(sigma600, 2) == 32.32
    assert round(sigma300, 2) == 16.16