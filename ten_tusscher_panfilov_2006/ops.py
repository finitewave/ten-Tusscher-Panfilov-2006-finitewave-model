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
    "get_diffusion_coefficient",
    "get_variables",
    "get_parameters",
    "ionic_step",
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
    "calc_d_inf",
    "calc_tau_f",
    "calc_f_inf",
    "calc_tau_f2",
    "calc_f2_inf",
    "calc_ical",
    "calc_r_inf",
    "calc_tau_s",
    "calc_s_inf",
    "calc_ito",
    "calc_xr1_inf",
    "calc_tau_xr1",
    "calc_xr2_inf",
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
)

from math import exp, log, sqrt


def get_diffusion_coefficient() -> float:
    """
    Returns the diffusion coefficient for spatial propagation in the model.
    """
    return {"D_model": 0.154}


def get_variables() -> dict[str, float]:
    """
    Returns default initial values for state variables.
    """
    return {
        "u": -84.5,  # Membrane potential (mV)
        "m": 0.0,  # Sodium activation gate
        "h": 0.75,  # Sodium inactivation gate
        "j": 0.75,  # Sodium inactivation gate
        "xr1": 0.0,  # Rapid delayed rectifier potassium activation gate
        "xs": 0.0,  # Slow delayed rectifier potassium activation gate
        "s": 1.0,  # Transient outward potassium inactivation gate
        "f": 1.0,  # L-type calcium channel inactivation gate
        "f2": 1.0,  # L-type calcium channel inactivation gate

    }


def get_parameters() -> dict[str, float]:
    """
    Returns default parameter values for the model.
    """
    return {
        "R": 8.314472,          # Gas constant (J/(mol*K))
        "T": 310.0,             # Absolute temperature (K)
        "F": 96.4853415,        # Faraday's constant (C/mol)
        "Ko": 5.4,              # Extracellular potassium concentration (mM)
        "Nao": 140.0,           # Extracellular sodium concentration (mM)
        "Cao": 2.0,             # Extracellular calcium concentration (mM)
        "nai": 7.67,            # Intracellular sodium concentration (mM)
        "ki": 138.3,            # Intracellular potassium concentration (mM)
        "cai": 0.00007,         # Intracellular calcium concentration (mM)

        "gNa": 14.838,          # Maximum conductance for fast sodium current (nS/pF)
        "gK1": 5.405,           # Maximum conductance for inward rectifier potassium current (nS/pF)
        "gto": 0.294,           # Maximum conductance for transient outward potassium current (nS/pF)
        "gKr": 0.101,           # Maximum conductance for rapid delayed rectifier potassium current (nS/pF)
        "gKs": 0.257,
        "pKNa": 0.03,           # Potassium-sodium exchange ratio for the slow delayed rectifier current
        "gCaL": 0.2786,        # Maximum conductance for L-type calcium current ([nS] * mm^3 / (ms * pF))
        "kNaCa": 1000.0,        # (pA / pF)
        "gamma": 0.35,          # Voltage dependence parameter for the sodium-calcium exchanger
        "KmCa": 1.38,           # (mM)
        "KmNai": 87.5,          # (mM)
        "ksat": 0.1,            # Saturation factor
        "alpha": 2.5,           # Scaling factor for the sodium-calcium exchanger current
        "pNaK": 2.724,          # (pA / pF)
        "KmK": 1.0,             # (mM)
        "KmNa": 40.0,           # (mM)
        "gpK": 0.0293,          # (nS/pF)
        "gpCa": 0.1238,
        "KpCa": 0.0005,
        "gbNa": 0.00029,
        "gbCa": 0.000592,
    }


def ionic_step(dt, u, cai, nai, ki, m, h, j, xr1, xs, s, f, f2, Ko, Cao, Nao, R, F, T,
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

    d_inf = calc_d_inf(u)

    f_inf = calc_f_inf(u)
    tau_f = calc_tau_f(u)
    f_new = calc_gating_variable_rush_larsen(f, f_inf, tau_f, dt)

    f2_inf = calc_f2_inf(u)
    tau_f2 = calc_tau_f2(u)
    f2_new = calc_gating_variable_rush_larsen(f2, f2_inf, tau_f2, dt)

    r_inf = calc_r_inf(u)

    s_inf = calc_s_inf(u)
    tau_s = calc_tau_s(u)
    s_new = calc_gating_variable_rush_larsen(s, s_inf, tau_s, dt)

    xr1_inf = calc_xr1_inf(u)
    tau_xr1 = calc_tau_xr1(u)
    xr1_new = calc_gating_variable_rush_larsen(xr1, xr1_inf, tau_xr1, dt)

    xr2_inf = calc_xr2_inf(u)

    xs_inf = calc_xs_inf(u)
    tau_xs = calc_tau_xs(u)
    xs_new = calc_gating_variable_rush_larsen(xs, xs_inf, tau_xs, dt)

    xk1_inf = calc_xk1_inf(u, Ek)

    ina = calc_ina(u, m, h, j, gNa, Ena)
    ibna = calc_ibna(u, Ena, gbNa)

    ito = calc_ito(u, r_inf, s, Ek, gto)
    ikr = calc_ikr(u, xr1, xr2_inf, Ek, gKr, Ko)
    iks = calc_iks(u, xs, Eks, gKs)
    ik1 = calc_ik1(u, Ek, gK1, xk1_inf, Ko)
    ipk = calc_ipk(u, Ek, gpK)

    ical = calc_ical(u, d_inf, f, f2, gCaL)
    ipca = calc_ipca(cai, KpCa, gpCa)
    ibca = calc_ibca(u, Eca, gbCa)

    inak = calc_inak(u, nai, Ko, KmK, KmNa, pNaK, F, R, T)
    inaca = calc_inaca(u, Nao, nai, Cao, cai, KmNai, KmCa, kNaCa, ksat, gamma, alpha, F, R, T)

    rhs = -calc_rhs(ikr, iks, ik1, ito, ina, ibna, ical, ibca, inak, inaca, ipca, ipk)

    return (rhs, m_new, h_new, j_new, xr1_new, xs_new, s_new, f_new, f2_new)


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

    Af2 = 600. * exp(-(u + 27.) * (u + 27.) / 170.)
    Bf2 = 7.75 / (1. + exp((25. - u) / 10.))
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

def calc_ical(u, d_inf, f, f2, gCaL):
    """
    Calculates the L-type calcium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    d_inf : np.ndarray
        Steady-state value of the gating variable d for L-type calcium channels.
    f : np.ndarray
        Gating variable for calcium-dependent calcium channels.
    f2 : np.ndarray
        Secondary gating variable for calcium-dependent calcium channels.
    gCaL : float
        Calcium conductance.

    Returns
    -------
    np.ndarray
        Updated L-type calcium current array.

    Note
    ----
    Singularity at u = 15 mV is handled by the limit as u approaches 15 mV,
    x  / (exp(x) - 1) = 1 as x approaches 0.
    """
    ical = gCaL * d_inf * f * f2 * (u - 60.)
    return ical


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


def calc_ito(u, r_inf, s, Ek, gto):
    """
    Calculates the transient outward current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    r_inf : np.ndarray
        Steady-state value of the gating variable r.
    s : np.ndarray
        Gating variable for calcium-sensitive current.
    ek : float
        Potassium reversal potential.

    Returns
    -------
    np.ndarray
        Updated transient outward current array.
    """

    return gto * r_inf * s * (u - Ek)


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


def calc_ikr(u, xr1, xr2_inf, Ek, gKr, Ko):
    """
    Calculates the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    xr1 : np.ndarray
        Gating variable for rapid delayed rectifier potassium channels.
    xr2_inf : np.ndarray
        Steady-state value of the gating variable xr2.
    Ek : float
        Potassium reversal potential.
    gKr : float
        Potassium conductance.

    Returns
    -------
    np.ndarray
        Updated rapid delayed rectifier potassium current array.
    """

    return gKr * sqrt(Ko / 5.4) * xr1 * xr2_inf * (u - Ek)


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


def calc_xk1_inf(u, Ek):
    """Calculates the steady-state value of the gating variable k1
    for the inward rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ek : float
        Potassium reversal potential.

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable k1.
    """
    axk1 = 0.1 / (1. + exp(0.06 * (u - Ek - 200.)))
    bxk1 = (3. * exp(0.0002 * (u - Ek + 100.)) + exp(0.1 * (u - Ek - 10.))) / (1. + exp(-0.5 * (u - Ek)))
    k1_inf = axk1 / (axk1 + bxk1)
    return k1_inf


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

def calc_ik1(u, Ek, gK1, xk1_inf, Ko):
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
    xk1_inf : float
        Steady-state value of the gating variable k1.
    Ko : float
        Potassium ion concentration in the extracellular space.

    Returns
    -------
    np.ndarray
        Updated inward rectifier potassium current array.
    """

    return gK1 * sqrt(Ko / 5.4) * xk1_inf * (u - Ek)

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

