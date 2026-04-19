"""
ops.py — mathematical core of the model.

This module provides functions to compute the model equations,
as well as functions to retrieve default parameters and initial
values for the state variables.

The TP06 model is a detailed biophysical model of the human ventricular 
action potential, designed to simulate realistic electrical behavior in 
tissue including alternans, reentrant waves, and spiral wave breakup.

References:
-----
ten Tusscher KH, Panfilov AV. 
Alternans and spiral breakup in a human ventricular tissue model.
Am J Physiol Heart Circ Physiol. 2006 Sep;291(3):H1088–H1100.
https://doi.org/10.1152/ajpheart.00109.2006
"""

__all__ = (
    "get_variables",
    "get_parameters",
    "calc_rhs",  
    "calc_where",
    "calc_gating_variable_rush_larsen",
    "calc_Ek",
    "calc_Ena",
    "calc_Eks",
    "calc_Eca",
    "calc_m_inf",
    "calc_tau_m",
    "calc_h_inf",
    "calc_tau_h",
    "calc_tau_j",
    "calc_ina",
    "calc_tau_d",
    "calc_d_inf",
    "calc_tau_f",
    "calc_f_inf",
    "calc_tau_f2",
    "calc_f2_inf",
    "calc_tau_fcass",
    "calc_fcass_inf",
    "calc_ical",
    "calc_tau_r",
    "calc_r_inf",
    "calc_tau_s",
    "calc_s_inf",
    "calc_ito",
    "calc_xr1_inf",
    "calc_tau_xr1",
    "calc_xr2_inf",
    "calc_tau_xr2",
    "calc_ikr",
    "calc_xs_inf",
    "calc_tau_xs",
    "calc_iks",
    "calc_ik1",
    "calc_inaca",
    "calc_inak",
    "calc_ipca",
    "calc_ipk",
    "calc_ibna",
    "calc_ibca",
    "calc_kCaSR",
    "calc_irel",
    "calc_ileak",
    "calc_iup",
    "calc_ixfer",
    "calc_casr",
    "calc_cass",
    "calc_cai",
    "calc_dnai",
    "calc_dki"
)

from math import exp, log, sqrt


def get_variables() -> dict[str, float]:
    """
    Returns default initial values for state variables.
    """
    return {
        "u": -84.5,  # Membrane potential (mV)
        "cai": 0.00007,  # Intracellular calcium concentration (mM)
        "casr": 1.3,  # SR calcium concentration (mM)
        "cass": 0.00007,  # Subspace calcium concentration (mM)
        "nai": 7.67,  # Intracellular sodium concentration (mM)
        "ki": 138.3,  # Intracellular potassium concentration (mM)
        "m": 0.0,  # Sodium activation gate
        "h": 0.75,  # Sodium inactivation gate
        "j": 0.75,  # Sodium inactivation gate
        "xr1": 0.0,  # Rapid delayed rectifier potassium activation gate
        "xr2": 1.0,  # Rapid delayed rectifier potassium activation gate
        "xs": 0.0,  # Slow delayed rectifier potassium activation gate
        "r": 0.0,  # Transient outward potassium activation gate
        "s": 1.0,  # Transient outward potassium inactivation gate
        "d": 0.0,  # L-type calcium channel activation gate
        "f": 1.0,  # L-type calcium channel inactivation gate
        "f2": 1.0,  # L-type calcium channel inactivation gate
        "fcass": 1.0,  # Calcium release inactivation gate
        "rr": 1.0,  # Ryanodine receptor activation gate

    }


def get_parameters() -> dict[str, float]:
    """
    Returns default parameter values for the model.
    """
    return {
        "R": 8.314472,          # Gas constant (J/(mol*K))
        "T": 310.0,             # Absolute temperature (K)
        "F": 96.4853415,        # Faraday's constant (C/mol)
        "Cm": 185.0,            # Membrane Cm (pF)
        "Vc": 16404.0,          # Cell volume (um^3)
        "Vsr": 1094.0,          # SR volume (um^3)
        "Vss": 54.68,           # Subspace volume (um^3)
        "Ko": 5.4,              # Extracellular potassium concentration (mM)
        "Nao": 140.0,           # Extracellular sodium concentration (mM)
        "Cao": 2.0,             # Extracellular calcium concentration (mM)
        "gNa": 14.838,          # Maximum conductance for fast sodium current (nS/pF)
        "gK1": 5.405,           # Maximum conductance for inward rectifier potassium current (nS/pF)
        "gto": 0.294,           # Maximum conductance for transient outward potassium current (nS/pF)
        "gKr": 0.153,           # Maximum conductance for rapid delayed rectifier potassium current (nS/pF)
        "gKs": 0.392,
        "pKNa": 0.03,           # Potassium-sodium exchange ratio for the slow delayed rectifier current
        "gCaL": 0.03980,        # Maximum conductance for L-type calcium current ([nS] * mm^3 / (ms * pF))
        "kNaCa": 1000.0,        # (pA / pF)
        "gamma": 0.35,          # Voltage dependence parameter for the sodium-calcium exchanger
        "KmCa": 1.38,           # (mM)
        "KmNai": 87.5,          # (mM)
        "ksat": 0.1,            # Saturation factor
        "alpha": 2.5,           # Scaling factor for the sodium-calcium exchanger current
        "pNaK": 2.724,          # (pA / pF)
        "KmK": 1.0,             # (mM)
        "KmNa": 40.0,           # (mM)
        "gpK": 0.0146,          # (nS/pF)
        "gpCa": 0.1238,
        "KpCa": 0.0005,
        "gbNa": 0.00029,
        "gbCa": 0.000592,
        "Vmaxup": 0.006375,
        "Kup": 0.00025,
        "Vrel": 0.102,
        "k1": 0.15,
        "k2": 0.045,
        "k3": 0.060,
        "k4": 0.005,
        "EC": 1.5,
        "maxsr": 2.5,
        "minsr": 1.0,
        "Vleak": 0.00036,
        "Vxfer": 0.0038,
        "Bufc": 0.2,            # Buffer concentration in cytosol (mM)
        "Kbufc": 0.001,         # Buffer dissociation constant in cytosol (mM)
        "Bufsr": 10.0,          # Buffer concentration in SR (mM)
        "Kbufsr": 0.3,
        "Bufss": 0.4,
        "Kbufss": 0.00025,
    }


def ionic_step(dt, u, cai, casr, cass, nai, ki, m, h, j, xr1, xr2, xs, r, s, d,
               f, f2, fcass, rr, Ko, Cao, Nao, Vc, Vsr, Vss, Bufc, Kbufc, 
               Bufsr, Kbufsr, Bufss, Kbufss, Vmaxup, Kup, Vrel, k1, k2, k3, k4,
               EC, maxsr, minsr, Vleak, Vxfer, R, F, T, Cm,
               gKr, gKs, gK1, gto, gNa, gbNa, gCaL, gbCa, gpCa, KpCa, gpK, pKNa,
               KmK, KmNa, pNaK, kNaCa, KmNai, KmCa, ksat, gamma, alpha):
    """
    Perform a single time step update.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    u : float
        Membrane potential variable.
    cai : float
        Intracellular calcium concentration variable.
    casr : float
        Sarcoplasmic reticulum calcium concentration variable.
    cass : float
        Subspace calcium concentration variable.
    nai : float
        Intracellular sodium concentration variable.
    ki : float
        Intracellular potassium concentration variable.
    m : float
        Sodium activation gate variable.
    h : float
        Sodium inactivation gate variable.
    j : float
        Sodium inactivation gate variable.
    xr1 : float
        Rapid delayed rectifier potassium activation gate variable.
    xr2 : float
        Rapid delayed rectifier potassium activation gate variable.
    xs : float
        Slow delayed rectifier potassium activation gate variable.
    r : float
        Transient outward potassium activation gate variable.
    s : float
        Transient outward potassium inactivation gate variable.
    d : float
        L-type calcium channel activation gate variable.
    f : float
        L-type calcium channel inactivation gate variable.
    f2 : float
        L-type calcium channel inactivation gate variable.
    fcass : float
        Calcium release inactivation gate variable.
    rr : float
        Ryanodine receptor activation gate variable.
    **parameters : float
        Model parameters.
    """

    Ek = calc_Ek(Ko, ki, R, T, F)
    Ena = calc_Ena(Nao, nai, R, T, F)
    Eks = calc_Eks(Ko, ki, Nao, nai, pKNa, R, T, F)
    Eca = calc_Eca(Cao, cai, R, T, F)

    m_inf = calc_m_inf(u)
    tau_m = calc_tau_m(u)
    m_new = calc_gating_variable_rush_larsen(m, m_inf, tau_m, dt)

    h_inf = calc_h_inf(u)
    tau_h = calc_tau_h(u)
    h_new = calc_gating_variable_rush_larsen(h, h_inf, tau_h, dt)

    j_inf = h_inf
    tau_j = calc_tau_j(u)
    j_new = calc_gating_variable_rush_larsen(j, j_inf, tau_j, dt)
    
    ina = calc_ina(u, m, h, j, gNa, Ena)

    d_inf = calc_d_inf(u)
    tau_d = calc_tau_d(u)
    d_new = calc_gating_variable_rush_larsen(d, d_inf, tau_d, dt)

    f_inf = calc_f_inf(u)
    tau_f = calc_tau_f(u)
    f_new = calc_gating_variable_rush_larsen(f, f_inf, tau_f, dt)

    f2_inf = calc_f2_inf(u)
    tau_f2 = calc_tau_f2(u)
    f2_new = calc_gating_variable_rush_larsen(f2, f2_inf, tau_f2, dt)

    fcass_inf = calc_fcass_inf(cass)
    tau_fcass = calc_tau_fcass(cass)
    fcass_new = calc_gating_variable_rush_larsen(fcass, fcass_inf, tau_fcass, dt)

    ical = calc_ical(u, d, f, f2, fcass, Cao, cass, gCaL, F, R, T)

    r_inf = calc_r_inf(u)
    tau_r = calc_tau_r(u)
    r_new = calc_gating_variable_rush_larsen(r, r_inf, tau_r, dt)

    s_inf = calc_s_inf(u)
    tau_s = calc_tau_s(u)
    s_new = calc_gating_variable_rush_larsen(s, s_inf, tau_s, dt)

    ito = calc_ito(u, r, s, Ek, gto)

    xr1_inf = calc_xr1_inf(u)
    tau_xr1 = calc_tau_xr1(u)
    xr1_new = calc_gating_variable_rush_larsen(xr1, xr1_inf, tau_xr1, dt)

    xr2_inf = calc_xr2_inf(u)
    tau_xr2 = calc_tau_xr2(u)
    xr2_new = calc_gating_variable_rush_larsen(xr2, xr2_inf, tau_xr2, dt)

    ikr = calc_ikr(u, xr1, xr2, Ek, gKr, Ko)

    xs_inf = calc_xs_inf(u)
    tau_xs = calc_tau_xs(u)
    xs_new = calc_gating_variable_rush_larsen(xs, xs_inf, tau_xs, dt)

    iks = calc_iks(u, xs, Eks, gKs)

    ik1 = calc_ik1(u, Ek, gK1)

    inaca = calc_inaca(u, Nao, nai, Cao, cai, KmNai, KmCa, kNaCa, ksat, gamma, alpha, F, R, T)

    inak = calc_inak(u, nai, Ko, KmK, KmNa, pNaK, F, R, T)

    ipca = calc_ipca(cai, KpCa, gpCa)

    ipk = calc_ipk(u, Ek, gpK)

    ibna = calc_ibna(u, Ena, gbNa)

    ibca = calc_ibca(u, Eca, gbCa)

    kCaSR = calc_kCaSR(casr, maxsr, minsr, EC)
    k1_ = k1 / kCaSR
    k2_ = k2 * kCaSR
    drr = k4 * (1 - rr) - k2_ * cass * rr
    rr_new = rr + dt * drr
    oo = k1_ * cass * cass * rr / (k3 + k1_ * cass * cass)

    irel = calc_irel(oo, casr, cass, Vrel)

    ileak = calc_ileak(casr, cai, Vleak)

    iup = calc_iup(cai, Vmaxup, Kup)

    ixfer = calc_ixfer(cass, cai, Vxfer)

    # Concentration updates from old state and old-state currents
    casr_new = calc_casr(dt, casr, Bufsr, Kbufsr, iup, irel, ileak)

    cass_new = calc_cass(dt, cass, Bufss, Kbufss, ixfer, irel, ical, Cm,
                         Vc, Vss, Vsr, F)

    cai_new = calc_cai(dt, cai, Bufc, Kbufc, ibca, ipca, inaca, iup, ileak, ixfer,
                       Cm, Vsr, Vc, F)

    dnai = calc_dnai(ina, ibna, inak, inaca, Cm, Vc, F)
    nai_new = nai + dt * dnai

    dki = calc_dki(ik1, ito, ikr, iks, inak, ipk, Cm, Vc, F)
    ki_new = ki + dt * dki

    rhs = -calc_rhs(ikr, iks, ik1, ito, ina, ibna, ical, ibca, inak, inaca,
                    ipca, ipk)

    return (rhs, cai_new, casr_new, cass_new, nai_new, ki_new, m_new, h_new, j_new,
            xr1_new, xr2_new, xs_new, r_new, s_new, d_new, f_new, f2_new, fcass_new, rr_new)


def calc_rhs(ikr, iks, ik1, ito, ina, ibna, ical, ibca, inak, inaca, ipca, ipk) -> float:
    """
    Computes the right-hand side of the model.

    Parameters
    ----------
    ikr : float
        Rapid delayed rectifier potassium current.
    iks : float
        Slow delayed rectifier potassium current.
    ik1 : float
        Inward rectifier potassium current.
    ito : float
        Transient outward potassium current.
    ina : float
        Fast sodium current.
    ibna : float
        Background sodium current.
    ical : float
        L-type calcium current.
    ibca : float
        Background calcium current.
    inak : float
        Sodium-potassium pump current.
    inaca : float
        Sodium-calcium exchanger current.
    ipca : float
        Calcium pump current.
    ipk : float
        Potassium pump current.
    """
    return ikr + iks + ik1 + ito + ina + ibna + ical + ibca + inak + inaca + ipca + ipk


def calc_where(cond, x, y):
    if cond:
        return x
    return y


def calc_gating_variable_rush_larsen(x, x_inf, tau_x, dt):
    """
    Calculates the gating variable using the Rush-Larsen method.

    Parameters
    ----------
    x : float
        Current value of the gating variable.
    x_inf : float
        Steady-state value of the gating variable.
    tau_x : float
        Time constant for the gating variable (ms).
    """

    return x_inf - (x_inf - x) * exp(-dt / tau_x)


def calc_Ek(Ko, ki, R, T, F):
    """
    Calculates the Nernst potential for potassium.

    Parameters
    ----------
    Ko : float
        Extracellular potassium concentration.
    ki : float
        Intracellular potassium concentration.
    R : float
        Gas constant.
    T : float
        Temperature.
    F : float
        Faraday's constant.

    Returns
    -------
    float
        Nernst potential for potassium.
    """

    return R * T / F * log(Ko / ki)


def calc_Ena(Nao, nai, R, T, F):
    """
    Calculates the Nernst potential for sodium.

    Parameters
    ----------
    Nao : float
        Extracellular sodium concentration.
    nai : float
        Intracellular sodium concentration.
    R : float
        Gas constant.
    T : float
        Temperature.
    F : float
        Faraday's constant.

    Returns
    -------
    float
        Nernst potential for sodium.  
    """

    return R * T / F * log(Nao / nai)


def calc_Eks(Ko, ki, Nao, nai, pKNa, R, T, F):
    """
    Calculates the Nernst potential for the slow delayed rectifier potassium current.

    Parameters
    ----------
    Ko : float
        Extracellular potassium concentration.
    ki : float
        Intracellular potassium concentration.
    Nao : float
        Extracellular sodium concentration.
    nai : float
        Intracellular sodium concentration.
    pKNa : float
        Relative permeability of sodium to potassium for the slow delayed rectifier current.
    R : float
        Gas constant.
    T : float
        Temperature.
    F : float
        Faraday's constant.

    Returns
    -------
    float
        Nernst potential for the slow delayed rectifier potassium current.
    """

    return R * T / F * log((Ko + pKNa * Nao) / (ki + pKNa * nai))


def calc_Eca(Cao, cai, R, T, F):
    """
    Calculates the Nernst potential for calcium.

    Parameters
    ----------
    Cao : float
        Extracellular calcium concentration.
    cai : float
        Intracellular calcium concentration.
    R : float
        Gas constant.
    T : float
        Temperature.
    F : float
        Faraday's constant.

    Returns
    -------
    float
        Gas constant times temperature divided by Faraday's constant.

    Returns
    -------
    float
        Nernst potential for calcium.
    """
    
    return 0.5 * R * T / F * log(Cao / cai)



def calc_m_inf(u):
    """
    Calculates the steady-state value of the gating variable m for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable m.
    """

    m_inf = 1. / ((1. + exp((-56.86 - u) / 9.03)) *
                  (1. + exp((-56.86 - u) / 9.03)))
    return m_inf


def calc_tau_m(u):
    """
    Calculates the time constant for the gating variable m for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Time constant for the gating variable m.
    """

    alpha_m = 1. / (1. + exp((-60. - u) / 5.))
    beta_m = 0.1 / (1. + exp((u + 35.) / 5.)) + 0.10 / (1. + exp((u - 50.) / 200.))
    tau_m = alpha_m * beta_m
    return tau_m


def calc_h_inf(u):
    """
    Calculates the steady-state value of the gating variable h for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable h.
    """

    h_inf = 1. / ((1. + exp((u + 71.55) / 7.43)) * (1. + exp((u + 71.55) / 7.43)))
    return h_inf


def calc_tau_h(u):
    """
    Calculates the time constant for the gating variable h for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Time constant for the gating variable h.
    """

    alpha_h = calc_where(u >= -40., 0, 0.057 * exp(-(u + 80.) / 6.8))
    beta_h = calc_where(u >= -40., 0.77 / (0.13 * (1. + exp(-(u + 10.66) / 11.1))),
                        2.7 * exp(0.079 * u) + (3.1e5) * exp(0.3485 * u))

    tau_h = 1.0 / (alpha_h + beta_h)
    return tau_h


def calc_tau_j(u):
    """
    Calculates the time constant for the gating variable j for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Time constant for the gating variable j.
    """

    # alpha_j = calc_where(u >= -40., 0.,
    #                      (-127140 * exp(0.2444 * u) - 3.474e-5 * exp(-0.04391 * u)) *
    #                      (u + 37.78) / (1 + exp(0.311 * (u + 79.23))))
    # beta_j = calc_where(u >= -40.,
    #                     0.6 * exp(-2.535e-7 * u) / (1 + exp(-0.1 * (u + 32))),
    #                     0.02424 * exp(-0.01052 * u) / (1 + exp(-0.1378 * (u + 40.14))))
    alpha_j = calc_where(u >= -40., 0.,
                         (-25428 * exp(0.2444 * u) - 6.948e-6 * exp(-0.04391 * u)) *
                         (u + 37.78) / (1 + exp(0.311 * (u + 79.23))))
    beta_j = calc_where(u >= -40.,
                        0.6 * exp(0.057 * u) / (1 + exp(-0.1 * (u + 32))),
                        0.02424 * exp(-0.01052 * u) / (1 + exp(-0.1378 * (u + 40.14))))
    tau_j = 1.0 / (alpha_j + beta_j)
    return tau_j


def calc_ina(u, m, h, j, gNa, Ena):
    """
    Calculates the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    m : np.ndarray
        Gating variable for sodium channels (activation).
    h : np.ndarray
        Gating variable for sodium channels (inactivation).
    j : np.ndarray
        Gating variable for sodium channels (inactivation).
    gNa : float
        Sodium conductance.
    Ena : float
        Sodium reversal potential.

    Returns
    -------
    np.ndarray
        Updated fast sodium current array.
    """

    return gNa * (m * m * m) * h * j * (u - Ena)


def calc_tau_d(u):
    """ 
    Calculates the time constant for the gating variable d for the L-type calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Time constant for the gating variable d.
    """

    Ad = 1.4 / (1. + exp((-35. - u) / 13.)) + 0.25
    Bd = 1.4 / (1. + exp((u + 5.) / 5.))
    Cd = 1.0 / (1. + exp((50. - u) / 20.))
    tau_d = Ad * Bd + Cd
    return tau_d


def calc_d_inf(u):
    """
    Calculates the steady-state value of the gating variable d for the L-type calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable d.
    """

    d_inf = 1. / (1. + exp((-8 - u) / 7.5))
    return d_inf


def calc_tau_f(u):
    """
    Calculates the time constant for the gating variable f for the L-type calcium current.
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Time constant for the gating variable f.
    """

    Af = 1102.5 * exp(-(u + 27) * (u + 27) / 225)
    Bf = 200. / (1. + exp((13. -  u) / 10.))
    Cf = (180. / (1. + exp((u + 30.) / 10.))) + 20.
    tau_f = Af + Bf + Cf
    return tau_f


def calc_f_inf(u):
    """
    Calculates the steady-state value of the gating variable f for the L-type calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable f.
    """

    f_inf = 1. / (1. + exp((u + 20.) / 7.))
    return f_inf


def calc_tau_f2(u):
    """
    Calculates the time constant for the secondary gating variable f2 for the L-type calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Time constant for the secondary gating variable f2.
    """

    Af2 = 600. * exp(-(u + 25.) * (u + 25.) / 170.)
    Bf2 = 31. / (1. + exp((25. - u) / 10.))
    Cf2 = 16. / (1. + exp((u + 30.) / 10.))
    # Af2 = 562. * exp(-(u + 27.) * (u + 27.) / 240.)
    # Bf2 = 31. / (1. + exp((25 - u) / 10.))
    # Cf2 = 80. / (1. + exp((u + 30) / 10.))
    tau_f2 = Af2 + Bf2 + Cf2
    return tau_f2


def calc_f2_inf(u):
    """
    Calculates the steady-state value of the secondary gating variable f2 for the L-type calcium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Steady-state value of the secondary gating variable f2.
    """

    f2_inf = 0.67 / (1. + exp((u + 35) / 7.43)) + 0.33
    return f2_inf


def calc_tau_fcass(cass):
    """
    Calculates the time constant for the gating variable fcass for the calcium-sensitive current.

    Parameters
    ----------
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable fcass.
    """

    tau_fcass = 80. / (1. + (cass / 0.05) * (cass / 0.05)) + 2.
    return tau_fcass


def calc_fcass_inf(cass):
    """
    Calculates the steady-state value of the gating variable fcass for the calcium-sensitive current.

    Parameters
    ----------
    cass : np.ndarray
        Calcium concentration in the submembrane space.

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable fcass.
    """
    
    fcass_inf = 0.6 / (1 + (cass / 0.05) * (cass / 0.05)) + 0.4
    return fcass_inf


def calc_ical(u, d, f, f2, fcass, Cao, cass, gCaL, F, R, T):
    """
    Calculates the L-type calcium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    d : np.ndarray
        Gating variable for L-type calcium channels.
    f : np.ndarray
        Gating variable for calcium-dependent calcium channels.
    f2 : np.ndarray
        Secondary gating variable for calcium-dependent calcium channels.
    fcass : np.ndarray
        Gating variable for calcium-sensitive current.
    Cao : float
        Extracellular calcium concentration.
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    gCaL : float
        Calcium conductance.
    F : float
        Faraday's constant.
    R : float
        Ideal gas constant.
    T : float
        Absolute temperature.
    Returns
    -------
    np.ndarray
        Updated L-type calcium current array.

    Note
    ----
    Singularity at u = 15 mV is handled by the limit as u approaches 15 mV,
    x  / (exp(x) - 1) = 1 as x approaches 0.
    """
    coeff = gCaL * d * f * f2 * fcass * 2 * F
    x = 2 * (u - 15) * F / (R * T)
    ical = calc_where(abs(u - 15) < 1e-6,
                      coeff * (0.25 * exp(x) * cass - Cao),
                      coeff * (0.25 * exp(x) * cass - Cao) * x / (exp(x) - 1.))
    return ical


def calc_tau_r(u):
    """Calculates the time constant for the gating variable r for the transient outward potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable r.
    """

    tau_r = 9.5 * exp(-(u + 40.) * (u + 40.) / 1800.) + 0.8
    return tau_r


def calc_r_inf(u):
    """Calculates the steady-state value of the gating variable r for the transient outward potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable r.
    """

    r_inf = 1. / (1. + exp((20. - u) / 6.))
    return r_inf


def calc_tau_s(u):
    """Calculates the time constant for the gating variable s for the transient outward potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable s.
    """

    tau_s = 85. * exp(-(u + 45.) * (u + 45.) / 320.) + 5. / (1. + exp((u - 20.) / 5.)) + 3.
    return tau_s


def calc_s_inf(u):
    """Calculates the steady-state value of the gating variable s for the transient outward potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable s.
    """

    s_inf = 1. / (1. + exp((u + 20.) / 5.))
    return s_inf


def calc_ito(u, r, s, Ek, gto):
    """
    Calculates the transient outward current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    r : np.ndarray
        Gating variable for ryanodine receptors.
    s : np.ndarray
        Gating variable for calcium-sensitive current.
    ek : float
        Potassium reversal potential.

    Returns
    -------
    np.ndarray
        Updated transient outward current array.
    """

    return gto * r * s * (u - Ek)


def calc_xr1_inf(u):
    """Calculates the steady-state value of the gating variable xr1 
    for the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable xr1.
    """

    xr1_inf = 1. / (1. + exp((-26. - u) / 7.))
    return xr1_inf


def calc_tau_xr1(u):
    """Calculates the time constant for the gating variable xr1 
    for the rapid delayed rectifier potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.

    Returns
    -------
    np.ndarray
        Time constant for the gating variable xr1.
    """

    axr1 = 450. / (1. + exp((-45. - u) / 10.))
    bxr1 = 6. / (1. + exp((u - (-30.)) / 11.5))
    tau_xr1 = axr1 * bxr1
    return tau_xr1


def calc_xr2_inf(u):
    """Calculates the steady-state value of the gating variable xr2 
    for the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable xr2.
    """

    xr2_inf = 1. / (1. + exp((u - (-88.)) / 24.))
    return xr2_inf


def calc_tau_xr2(u):
    """Calculates the time constant for the gating variable xr2 
    for the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable xr2.
    """

    axr2 = 3. / (1. + exp((-60. - u) / 20.))
    bxr2 = 1.12 / (1. + exp((u - 60.) / 20.))
    tau_xr2 = axr2 * bxr2
    return tau_xr2


def calc_ikr(u, xr1, xr2, Ek, gKr, Ko):
    """
    Calculates the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    xr1 : np.ndarray
        Gating variable for rapid delayed rectifier potassium channels.
    xr2 : np.ndarray
        Gating variable for rapid delayed rectifier potassium channels.
    Ek : float
        Potassium reversal potential.
    gKr : float
        Potassium conductance.

    Returns
    -------
    np.ndarray
        Updated rapid delayed rectifier potassium current array.
    """

    return gKr * sqrt(Ko / 5.4) * xr1 * xr2 * (u - Ek)


def calc_xs_inf(u):
    """Calculates the steady-state value of the gating variable xs 
    for the slow delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable xs.
    """

    xs_inf = 1. / (1. + exp((-5. - u) / 14.))
    return xs_inf


def calc_tau_xs(u):
    """Calculates the time constant for the gating variable xs
    for the slow delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable xs.
    """
    
    Axs = 1400. / (sqrt(1. + exp((5. - u) / 6.)))
    Bxs = 1. / (1. + exp((u - 35.) / 15.))
    tau_xs = Axs * Bxs + 80.
    return tau_xs


def calc_iks(u, xs, Eks, gKs):
    """
    Calculates the slow delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    xs : np.ndarray
        Gating variable for slow delayed rectifier potassium channels.
    Eks : float
        Potassium reversal potential.
    gKs : float
        Potassium conductance.
    
    Returns
    -------
    np.ndarray
        Updated slow delayed rectifier potassium current array.
    """

    return gKs * xs * xs * (u - Eks)

def calc_ik1(u, Ek, gK1):
    """
    Calculates the inward rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ek : float
        Potassium reversal potential.
    gK1 : float
        Inward rectifier potassium conductance.

    Returns
    -------
    np.ndarray
        Updated inward rectifier potassium current array.
    """

    ak1 = 0.1 / (1. + exp(0.06 * (u - Ek - 200.)))
    bk1 = (3. * exp(0.0002 * (u - Ek + 100.)) +
           exp(0.1 * (u - Ek - 10.)) / (1. + exp(-0.5 * (u - Ek))))
    rec_iK1 = ak1 / (ak1 + bk1)

    return gK1 * rec_iK1 * (u - Ek)

def calc_inaca(u, Nao, nai, Cao, cai, KmNai, KmCa, kNaCa, ksat, gamma, alpha, F, R, T):
    """
    Calculates the sodium-calcium exchanger current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Nao : float
        Sodium ion concentration in the extracellular space.
    nai : np.ndarray
        Sodium ion concentration in the intracellular space.
    Cao : float
        Calcium ion concentration in the extracellular space.
    cai : np.ndarray
        Calcium ion concentration in the submembrane space.
    KmNai : float
        Michaelis constant for sodium.
    KmCa : float
        Michaelis constant for calcium.
    kNaCa : float
        Sodium-calcium exchanger conductance.
    ksat : float
        Saturation factor.
    gamma : float
        Exponent for sodium dependence.
    alpha : float
        Scaling factor.
    F : float
        Faraday's constant.
    R : float
        Ideal gas constant.
    T : float
        Temperature.
    
    Returns
    -------
    np.ndarray
        Updated sodium-calcium exchanger current array.
    """

    inaca = (kNaCa * (1. / (KmNai * KmNai * KmNai + Nao * Nao * Nao)) * (1. / (KmCa + Cao)) *
             (1. / (1 + ksat * exp((gamma - 1) * u * F / (R * T)))) *
             (exp(gamma * u * F / (R * T)) * nai * nai * nai * Cao -
              exp((gamma - 1) * u * F / (R * T)) * Nao * Nao * Nao * cai * alpha))
    return inaca

def calc_inak(u, nai, Ko, KmK, KmNa, pNaK, F, R, T):
    """
    Calculates the sodium-potassium pump current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    nai : np.ndarray
        Sodium ion concentration in the intracellular space.
    Ko : float
        Potassium ion concentration in the extracellular space.
    KmK : float
        Michaelis constant for potassium.
    KmNa : float
        Michaelis constant for sodium.
    pNaK : float
        Sodium-potassium pump conductance.
    F : float
        Faraday's constant.
    R : float
        Ideal gas constant.
    T : float
        Temperature.

    Returns
    -------
    np.ndarray
        Updated sodium-potassium pump current array.
    """

    rec_iNaK = (1. / (1. + 0.1245 * exp(-0.1 * u * F / (R * T)) + 
                      0.0353 * exp(-u * F / (R * T))))

    return pNaK * (Ko / (Ko + KmK)) * (nai / (nai + KmNa)) * rec_iNaK

def calc_ipca(cai, KpCa, gpCa):
    """
    Calculates the calcium pump current.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    KpCa : float
        Michaelis constant for calcium pump.
    gpCa : float
        Calcium pump conductance.

    Returns
    -------
    np.ndarray
        Updated calcium pump current array.
    """

    return gpCa * cai / (KpCa + cai)

def calc_ipk(u, Ek, gpK):
    """
    Calculates the potassium pump current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ek : float
        Potassium reversal potential.
    gpK : float
        Potassium pump conductance.
    
    Returns
    -------
    np.ndarray
        Updated potassium pump current array.
    """
    rec_ipK = 1. / (1. + exp((25 - u) / 5.98))

    return gpK * rec_ipK * (u - Ek)

def calc_ibna(u, Ena, gbNa):
    """
    Calculates the background sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ena : float
        Sodium reversal potential.
    gbNa : float
        Background sodium conductance.

    Returns
    -------
    np.ndarray
        Updated background sodium current array.
    """

    return gbNa * (u - Ena)

def calc_ibca(u, Eca, gbCa):
    """
    Calculates the background calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Eca : float
        Calcium reversal potential.
    gbCa : float
        Background calcium conductance.

    Returns
    -------
    np.ndarray
        Updated background calcium current array.
    """

    return gbCa * (u - Eca)

def calc_kCaSR(casr, maxsr, minsr, EC):
    """
    Calculates the kCaSR for the ryanodine receptor.

    Parameters
    ----------
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    maxsr : float
        Maximum value of kcasr.
    minsr : float
        Minimum value of kcasr.
    EC : float
        CaSR half-saturation constant of kcasr.

    Returns
    -------
    np.ndarray
        kCaSR
    """

    return maxsr - ((maxsr - minsr) / (1 + (EC / casr) * (EC / casr)))

def calc_irel(oo, casr, cass, vrel):
    """
    Calculates the ryanodine receptor current.

    Parameters
    ----------
    oo : np.ndarray
        Ryanodine receptor gating variable for calcium release.
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    vrel : float
        Release rate of calcium from the sarcoplasmic reticulum.
    
    Returns
    -------
    np.ndarray
        Updated ryanodine receptor current array.
    """

    return vrel * oo * (casr - cass)

def calc_ileak(casr, cai, vleak):
    """
    Calculates the calcium leak current.

    Parameters
    ----------
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    vleak : float
        Leak rate of calcium from the sarcoplasmic reticulum.

    Returns
    -------
    np.ndarray
        Updated calcium leak current array.
    """

    return vleak * (casr - cai)

def calc_iup(cai, vmaxup, Kup):
    """
    Calculates the calcium uptake current.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    vmaxup : float
        Uptake rate of calcium into the sarcoplasmic reticulum.
    Kup : float
        Michaelis constant for calcium uptake.

    Returns
    -------
    np.ndarray
        Updated calcium uptake current array.
    """

    return vmaxup / (1. + ((Kup * Kup) / (cai * cai)))

def calc_ixfer(cass, cai, vxfer):
    """
    Calculates the calcium transfer current.

    Parameters
    ----------
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    vxfer : float
        Transfer rate of calcium between the submembrane space and cytosol.

    Returns
    -------
    np.ndarray
        Updated calcium transfer current array.
    """

    return vxfer * (cass - cai)

def calc_casr(dt, caSR, bufsr, Kbufsr, iup, irel, ileak):
    """
    Calculates the calcium concentration in the sarcoplasmic reticulum.

    Parameters
    ----------
    casr : np.ndarray
        Calcium concentration in the sarcoplasmic reticulum.
    bufsr : float
        Buffering capacity of the sarcoplasmic reticulum.
    Kbufsr : float
        Buffering constant of the sarcoplasmic reticulum.
    iup : float
        Calcium uptake current.
    irel : float
        Calcium release current.
    ileak : float
        Leak rate of calcium from the sarcoplasmic reticulum.

    Returns
    -------
    np.ndarray
        Updated calcium concentration in the sarcoplasmic reticulum.
    """

    CaCSQN = bufsr * caSR / (caSR + Kbufsr)
    dCaSR = dt * (iup - irel - ileak)
    casr_total = CaCSQN + dCaSR + caSR
    bjsr = bufsr + Kbufsr - casr_total
    cjsr = Kbufsr * casr_total
    return (sqrt(bjsr * bjsr + 4 * cjsr) - bjsr) / 2

def calc_cass(dt, caSS, bufss, Kbufss, ixfer, irel, ical, Cm, Vc, Vss, Vsr, F):
    """
    Calculates the calcium concentration in the submembrane space.

    Parameters
    ----------
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    bufss : float
        Buffering capacity of the submembrane space.
    Kbufss : float
        Buffering constant of the submembrane space.
    ixfer : float
        Calcium transfer current.
    irel : float
        Calcium release current.
    ical : float
        L-type calcium current.
    Cm : float
        Membrane Cm.
    Vc : float
        Volume of the cytosol.
    Vss : float
        Volume of the submembrane space.
    Vsr : float
        Volume of the sarcoplasmic reticulum.
    F : float
        Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated calcium concentration in the submembrane space.
    """

    CaSSBuf = bufss * caSS / (caSS + Kbufss)
    dCaSS = dt * (-ixfer * (Vc / Vss) + irel * (Vsr / Vss) +
                  (-ical * Cm / (2 * Vss * F)))
    cass_total = CaSSBuf + dCaSS + caSS
    bcss = bufss + Kbufss - cass_total
    ccss = Kbufss * cass_total
    return (sqrt(bcss * bcss + 4 * ccss) - bcss) / 2

def calc_cai(dt, cai, bufc, Kbufc, ibca, ipca, inaca, iup, ileak, ixfer,
             Cm, Vsr, Vc, F):
    """
    Calculates the calcium concentration in the cytosol.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the cytosol.
    bufc : float
        Buffering capacity of the cytosol.
    Kbufc : float
        Buffering constant of the cytosol.
    ibca : float
        Background calcium current.
    ipca : float
        Calcium pump current.
    inaca : float
        Sodium-calcium exchanger current.
    iup : float
        Calcium uptake current.
    ileak : float
        Calcium leak current.
    ixfer : float
        Calcium transfer current.
    Cm : float
        Membrane Cm.
    Vsr : float
        Volume of the sarcoplasmic reticulum.
    Vc : float
        Volume of the cytosol.
    F : float
        Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated calcium concentration in the cytosol.
    """

    CaCBuf = bufc * cai / (cai + Kbufc)
    dCai = dt * ((-(ibca + ipca - 2 * inaca) * Cm / (2 * Vc * F)) -
                   (iup - ileak) * (Vsr / Vc) + ixfer)
    cai_total = CaCBuf + dCai + cai
    bc = bufc + Kbufc - cai_total
    cc = Kbufc * cai_total
    return (-bc + sqrt(bc * bc + 4 * cc)) / 2

def calc_dnai(ina, ibna, inak, inaca, Cm, Vc, F):
    """
    Calculates the sodium concentration in the cytosol.

    Parameters
    ----------
    ina : float
        Fast sodium current.
    ibna : float
        Background sodium current.
    inak : float
        Sodium-potassium pump current.
    inaca : float
        Sodium-calcium exchanger current.
    Cm : float
        Membrane Cm.
    Vc : float
        Volume of the cytosol.
    F : float
        Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated sodium concentration in the cytosol.
    """

    dNai = -(ina + ibna + 3 * inak + 3 * inaca) * Cm / (Vc * F)
    return dNai

def calc_dki(ik1, ito, ikr, iks, inak, ipk, Cm, Vc, F):
    """
    Calculates the potassium concentration in the cytosol.

    Parameters
    ----------
    ik1 : float
        Inward rectifier potassium current.
    ito : float
        Transient outward current.
    ikr : float
        Rapid delayed rectifier potassium current.
    iks : float
        Slow delayed rectifier potassium current.
    inak : float
        Sodium-potassium pump current.
    ipk : float
        Potassium pump current.
    Cm : float
        Membrane Cm.
    Vc : float
        Volume of the cytosol.
    F : float
        Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated potassium concentration in the cytosol.
    """

    dKi = -(ik1 + ito + ikr + iks - 2 * inak + ipk) * Cm / (Vc * F)
    return dKi
