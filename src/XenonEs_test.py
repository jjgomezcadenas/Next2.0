from numpy.testing import assert_almost_equal
import numpy as np
from pytest import fixture

from pyNext.system_of_units import *
from Next2.src.XenonES import XenonES

@fixture(scope='module')
def xe():
    """returns instance of XenonES. """

    xe = XenonES()
    return xe

def test_pressure_at_t0_rho0_close_to_1atm(xe):
    """RHO is the critical density for which P = 1 atm for T = 0 C"""

    T0 = 273.15 # 0 in K
    assert_almost_equal(xe.P(T=0, RHO=5.9 * kg/m3, perfect=False) / atmosphere, 1, decimal=4)
    assert_almost_equal(xe.P2(T=0, RHO=5.9 * kg/m3, perfect=False) / atmosphere, 1, decimal=4)
    assert_almost_equal(xe.P(T=T0, RHO=5.9 * kg/m3, perfect=False, temp='K') / atmosphere,
                        1, decimal=4)
    assert_almost_equal(xe.P(T=T0, RHO=5.9 * kg/m3, perfect=False, temp='K') / MPa,
                    0.1, decimal=2)
    assert_almost_equal(xe.P(T=0, RHO=5.9 * kg/m3, perfect=True) / atmosphere, 1, decimal=2)
    assert_almost_equal(xe.P2(T=0, RHO=5.9 * kg/m3, perfect=True) / atmosphere, 1, decimal=2)


def test_pressure_at_T_20_rho2020_close_to_20atm(xe):
    rho_2020 = 124.3 * kg/m3
    assert_almost_equal(xe.P(T=20, RHO=rho_2020, perfect=False) / atmosphere, 20, decimal=1)
    assert_almost_equal(xe.P2(T=20, RHO=rho_2020, perfect=False) / atmosphere, 20, decimal=1)

def test_pressure_at_T_30_rho2030_close_to_30atm(xe):
    rho_3020 = 203.35 * kg/m3
    assert_almost_equal(xe.P(T=20, RHO=rho_3020, perfect=False) / atmosphere, 30, decimal=1)
    assert_almost_equal(xe.P2(T=20, RHO=rho_3020, perfect=False) / atmosphere, 29.6, decimal=1)

def test_pressure_at_T_15_rho2015_close_to_15atm(xe):
    rho_1520 = 89.9 * kg/m3
    assert_almost_equal(xe.P(T=20, RHO=rho_1520, perfect=False) / atmosphere, 15, decimal=1)
    assert_almost_equal(xe.P2(T=20, RHO=rho_1520, perfect=False) / atmosphere, 15, decimal=1)

def test_pressure_at_T_20_rho2010_close_to_10atm(xe):
    rho_1020 = 58 * kg/m3
    assert_almost_equal(xe.P(T=20, RHO=rho_1020, perfect=False) / atmosphere, 10, decimal=1)
    assert_almost_equal(xe.P2(T=20, RHO=rho_1020, perfect=False) / atmosphere, 10, decimal=1)

def test_pressure_compared_with_tabulated_value(xe):
    T = [248.400, 222.97, 200.270, 180.01, 224.98]
    RHO = np.array([0.01747, 0.01751, 0.01753, 0.01755, 0.09244]) * g/cm3
    P = np.array([0.2683, 0.24, 0.2143, 0.1911, 1.1163]) * MPa
    for t,rho,p in zip(T,RHO,P):
        assert_almost_equal(xe.P(T=t, RHO=rho, perfect=False, temp='K') / atmosphere,
                            p / atmosphere, decimal=2)
