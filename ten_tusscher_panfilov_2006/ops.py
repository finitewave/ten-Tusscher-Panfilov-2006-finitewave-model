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
    "calc_gating_variable_rush_larsen",
    "calc_gating_m",
    "calc_gating_h",
    "calc_gating_j",
    "calc_ina",
    "calc_ical",
    "calc_ito",
    "calc_ikr",
    "calc_iks",
    "calc_ik1",
    "calc_inaca",
    "calc_inak",
    "calc_ipca",
    "calc_ipk",
    "calc_ibna",
    "calc_ibca",
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

import math


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
        "Ki": 138.3,  # Intracellular potassium concentration (mM)
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
        "oo": 0.0,  # Ryanodine receptor open probability

    }


def get_parameters() -> dict[str, float]:
    """
    Returns default parameter values for the model.
    """
    return {
      "ko": 5.4,
        "cao": 2.0,
        "nao": 140.0,
        "Vc": 0.016404,
        "Vsr": 0.001094,
        "Vss": 0.00005468,
        "Bufc": 0.2,
        "Kbufc": 0.001,
        "Bufsr": 10.0,
        "Kbufsr": 0.3,
        "Bufss": 0.4,
        "Kbufss": 0.00025,
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
        "R": 8314.472,
        "F": 96485.3415,
        "T": 310.0,
        "RTONF": 26.71376,
        "CAPACITANCE": 0.185,
        "gkr": 0.153,
        "gks": 0.392,
        "gk1": 5.405,
        "gto": 0.294,
        "gna": 14.838,
        "gbna": 0.00029,
        "gcal": 0.00003980,
        "gbca": 0.000592,
        "gpca": 0.1238,
        "KpCa": 0.0005,
        "gpk": 0.0146,
        "pKNa": 0.03,
        "KmK": 1.0,
        "KmNa": 40.0,
        "knak": 2.724,
        "knaca": 1000,
        "KmNai": 87.5,
        "KmCa": 1.38,
        "ksat": 0.1,
        "n_": 0.35,
    }


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


def calc_gating_variable_rush_larsen(x, x_inf, tau_x, dt, exp=math.exp):
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
    exp : callable
        Exponential function to use (default: math.exp).
    """

    return x_inf - (x_inf - x)*exp(-dt/tau_x)


def calc_Ek(ko, ki, RTONF):
    """
    Calculates the Nernst potential for potassium.

    Parameters
    ----------
    ko : float
        Extracellular potassium concentration.
    ki : float
        Intracellular potassium concentration.
    RTONF : float
        Gas constant times temperature divided by Faraday's constant.

    Returns
    -------
    float
        Nernst potential for potassium.
    """

    return RTONF * math.log(ko / ki)


def calc_Ena(nao, nai, RTONF):
    """
    Calculates the Nernst potential for sodium.

    Parameters
    ----------
    nao : float
        Extracellular sodium concentration.
    nai : float
        Intracellular sodium concentration.
    RTONF : float
        Gas constant times temperature divided by Faraday's constant.

    Returns
    -------
    float
        Nernst potential for sodium.  
    """

    return RTONF * math.log(nao / nai)


def calc_Eks(ko, ki, nao, nai, pKNa, RTONF):
    """
    Calculates the Nernst potential for the slow delayed rectifier potassium current.

    Parameters
    ----------
    ko : float
        Extracellular potassium concentration.
    ki : float
        Intracellular potassium concentration.
    nao : float
        Extracellular sodium concentration.
    nai : float
        Intracellular sodium concentration.
    pKNa : float
        Relative permeability of sodium to potassium for the slow delayed rectifier current.
    RTONF : float
        Gas constant times temperature divided by Faraday's constant.

    Returns
    -------
    float
        Nernst potential for the slow delayed rectifier potassium current.
    """

    return RTONF * math.log((ko + pKNa * nao) / (ki + pKNa * nai))


def calc_Eca(cao, cai, RTONF):
    """
    Calculates the Nernst potential for calcium.

    Parameters
    ----------
    cao : float
        Extracellular calcium concentration.
    cai : float
        Intracellular calcium concentration.
    RTONF : float
        Gas constant times temperature divided by Faraday's constant.

    Returns
    -------
    float
        Nernst potential for calcium.
    """
    
    return 0.5 * RTONF * math.log(cao / cai)



def calc_m_inf(u, exp=math.exp):
    """
    Calculates the steady-state value of the gating variable m for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable m.
    """

    m_inf = 1./((1.+exp((-56.86-u)/9.03))
                * (1.+exp((-56.86-u)/9.03)))
    return m_inf


def calc_tau_m(u, exp=math.exp):
    """
    Calculates the time constant for the gating variable m for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Time constant for the gating variable m.
    """

    alpha_m = 1./(1.+exp((-60.-u)/5.))
    beta_m = 0.1/(1.+exp((u+35.)/5.)) + \
        0.10/(1.+exp((u-50.)/200.))
    tau_m = alpha_m*beta_m
    return tau_m


def calc_h_inf(u, exp=math.exp, where=calc_where):
    """
    Calculates the steady-state value of the gating variable h for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    where : callable
        Function to use for conditional logic (default: calc_where).

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable h.
    """

    h_inf = 1./((1.+exp((u+71.55)/7.43))
                * (1.+exp((u+71.55)/7.43)))
    return h_inf


def calc_tau_h(u, exp=math.exp, where=calc_where):
    """
    Calculates the time constant for the gating variable h for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    where : callable
        Function to use for conditional logic (default: calc_where).

    Returns
    -------
    np.ndarray
        Time constant for the gating variable h.
    """

    alpha_h = where(u >= -40., 0, 0.057*exp(-(u+80.)/6.8))
    beta_h = where(u >= -40., 0.77/(0.13*(1.+exp(-(u+10.66)/11.1))),
                    2.7*exp(0.079*u)+(3.1e5)*exp(0.3485*u))

    tau_h = 1.0/(alpha_h + beta_h)
    return tau_h


def calc_tau_j(u, exp=math.exp, where=calc_where):
    """
    Calculates the time constant for the gating variable j for the fast sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    where : callable
        Function to use for conditional logic (default: calc_where).

    Returns
    -------
    np.ndarray
        Time constant for the gating variable j.
    """

    alpha_j = where(u >= -40., 0, (-127140*exp(0.2444*u) - 3.474e-5*exp(-0.04391*u)) * \
                    (u+37.78) / (1+exp(0.311*(u+79.23))))
    beta_j = where(u >= -40., 0.6*exp(-2.535e-7*u) / (1+exp(-0.1*(u+32))),
                    0.02424*exp(-0.01052*u) / (1+exp(-0.1378*(u+40.14))))

    tau_j = 1.0/(alpha_j + beta_j)
    return tau_j


def calc_ina(u, m, h, j, gna, Ena):
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
    gna : float
        Sodium conductance.
    Ena : float
        Sodium reversal potential.

    Returns
    -------
    np.ndarray
        Updated fast sodium current array.
    """

    return gna*(m**3)*h*j*(u-Ena)


def calc_tau_d(u, exp=math.exp):
    """ 
    Calculates the time constant for the gating variable d for the L-type calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Time constant for the gating variable d.
    """

    Ad = 1.4/(1.+exp((-35-u)/13))+0.25
    Bd = 1.4/(1.+exp((u+5)/5))
    Cd = 1./(1.+exp((50-u)/20))
    tau_d = Ad*Bd+Cd
    return tau_d


def calc_d_inf(u, exp=math.exp):
    """
    Calculates the steady-state value of the gating variable d for the L-type calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable d.
    """

    d_inf = 1./(1.+exp((-8-u)/7.5))
    return d_inf


def calc_tau_f(u, exp=math.exp):
    """
    Calculates the time constant for the gating variable f for the L-type calcium current.
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Time constant for the gating variable f.
    """

    Af = 1102.5*exp(-(u+27)*(u+27)/225)
    Bf = 200./(1+exp((13-u)/10.))
    Cf = (180./(1+exp((u+30)/10)))+20
    tau_f = Af+Bf+Cf
    return tau_f


def calc_f_inf(u, exp=math.exp):
    """
    Calculates the steady-state value of the gating variable f for the L-type calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable f.
    """

    f_inf = 1./(1.+exp((u+20)/7))
    return f_inf


def calc_tau_f2(u, exp=math.exp):
    """
    Calculates the time constant for the secondary gating variable f2 for the L-type calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Time constant for the secondary gating variable f2.
    """

    Af2 = 600*exp(-(u+25)*(u+25)/170)
    Bf2 = 31/(1.+exp((25-u)/10))
    Cf2 = 16/(1.+exp((u+30)/10))
    tau_f2 = Af2+Bf2+Cf2
    return tau_f2


def calc_f2_inf(u, exp=math.exp):
    """
    Calculates the steady-state value of the secondary gating variable f2 for the L-type calcium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Steady-state value of the secondary gating variable f2.
    """

    f2_inf = 0.67/(1.+exp((u+35)/7.43))+0.33
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

    tau_fcass = 80./(1+(cass/0.05)*(cass/0.05))+2.
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
    
    fcass_inf = 0.6/(1+(cass/0.05)*(cass/0.05))+0.4
    return fcass_inf


def calc_ical(u, d, f, f2, fcass, cao, cass, gcal, F, R, T, exp=math.exp):
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
    cao : float
        Extracellular calcium concentration.
    cass : np.ndarray
        Calcium concentration in the submembrane space.
    gcal : float
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
    """

    return gcal*d*f*f2*fcass*4*(u-15)*(F*F/(R*T)) *\
        (0.25*exp(2*(u-15)*F/(R*T))*cass-cao) / \
        (exp(2*(u-15)*F/(R*T))-1.)


def calc_tau_r(u, exp=math.exp):
    """Calculates the time constant for the gating variable r for the transient outward potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable r.
    """

    tau_r = 9.5*exp(-(u+40.)*(u+40.)/1800.)+0.8
    return tau_r


def calc_r_inf(u, exp=math.exp):
    """Calculates the steady-state value of the gating variable r for the transient outward potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable r.
    """

    r_inf = 1./(1.+exp((20-u)/6.))
    return r_inf


def calc_tau_s(u, exp=math.exp):
    """Calculates the time constant for the gating variable s for the transient outward potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable s.
    """

    tau_s = 85.*exp(-(u+45.)*(u+45.)/320.) + 5./(1.+exp((u-20.)/5.))+3.
    return tau_s


def calc_s_inf(u, exp=math.exp):
    """Calculates the steady-state value of the gating variable s for the transient outward potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable s.
    """

    s_inf = 1./(1.+exp((u+20)/5.))
    return s_inf


def calc_ito(u, r, s, Ek, gto, exp=math.exp):
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

    return gto*r*s*(u-Ek)


def calc_xr1_inf(u, exp=math.exp):
    """Calculates the steady-state value of the gating variable xr1 
    for the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable xr1.
    """

    xr1_inf = 1./(1.+exp((-26.-u)/7.))
    return xr1_inf


def calc_tau_xr1(u, exp=math.exp):
    """Calculates the time constant for the gating variable xr1 
    for the rapid delayed rectifier potassium current.
    
    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).

    Returns
    -------
    np.ndarray
        Time constant for the gating variable xr1.
    """

    axr1 = 450./(1.+exp((-45.-u)/10.))
    bxr1 = 6./(1.+exp((u-(-30.))/11.5))
    tau_xr1 = axr1*bxr1
    return tau_xr1


def calc_xr2_inf(u, exp=math.exp):
    """Calculates the steady-state value of the gating variable xr2 
    for the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable xr2.
    """

    xr2_inf = 1./(1.+exp((u-(-88.))/24.))
    return xr2_inf


def calc_tau_xr2(u, exp=math.exp):
    """Calculates the time constant for the gating variable xr2 
    for the rapid delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable xr2.
    """

    axr2 = 3./(1.+exp((-60.-u)/20.))
    bxr2 = 1.12/(1.+exp((u-60.)/20.))
    tau_xr2 = axr2*bxr2
    return tau_xr2


def calc_ikr(u, xr1, xr2, Ek, gkr, ko, exp=math.exp, sqrt=math.sqrt):
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
    gkr : float
        Potassium conductance.

    Returns
    -------
    np.ndarray
        Updated rapid delayed rectifier potassium current array.
    """

    return gkr*sqrt(ko/5.4)*xr1*xr2*(u-Ek)


def calc_xs_inf(u, exp=math.exp):
    """Calculates the steady-state value of the gating variable xs 
    for the slow delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    
    Returns
    -------
    np.ndarray
        Steady-state value of the gating variable xs.
    """

    xs_inf = 1./(1.+exp((-5.-u)/14.))
    return xs_inf


def calc_tau_xs(u, exp=math.exp, sqrt=math.sqrt):
    """Calculates the time constant for the gating variable xs
    for the slow delayed rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    exp : callable
        Exponential function to use (default: math.exp).
    sqrt : callable
        Square root function to use (default: math.sqrt).
    
    Returns
    -------
    np.ndarray
        Time constant for the gating variable xs.
    """
    
    Axs = (1400./(sqrt(1.+exp((5.-u)/6))))
    Bxs = (1./(1.+exp((u-35.)/15.)))
    tau_xs = Axs*Bxs+80
    return tau_xs


def calc_iks(u, xs, Eks, gks):
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
    gks : float
        Potassium conductance.
    
    Returns
    -------
    np.ndarray
        Updated slow delayed rectifier potassium current array.
    """

    return gks*xs*xs*(u-Eks)

def calc_ik1(u, Ek, gk1, exp=math.exp):
    """
    Calculates the inward rectifier potassium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ek : float
        Potassium reversal potential.
    gk1 : float
        Inward rectifier potassium conductance.

    Returns
    -------
    np.ndarray
        Updated inward rectifier potassium current array.
    """

    ak1 = 0.1/(1.+exp(0.06*(u-Ek-200)))
    bk1 = (3.*exp(0.0002*(u-Ek+100)) +
           exp(0.1*(u-Ek-10)))/(1.+exp(-0.5*(u-Ek)))
    rec_iK1 = ak1/(ak1+bk1)

    return gk1*rec_iK1*(u-Ek)

def calc_inaca(u, nao, nai, cao, cai, KmNai, KmCa, knaca, ksat, n_, F, R, T, exp=math.exp):
    """
    Calculates the sodium-calcium exchanger current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    nao : float
        Sodium ion concentration in the extracellular space.
    nai : np.ndarray
        Sodium ion concentration in the intracellular space.
    cao : float
        Calcium ion concentration in the extracellular space.
    cai : np.ndarray
        Calcium ion concentration in the submembrane space.
    KmNai : float
        Michaelis constant for sodium.
    KmCa : float
        Michaelis constant for calcium.
    knaca : float
        Sodium-calcium exchanger conductance.
    ksat : float
        Saturation factor.
    n_ : float
        Exponent for sodium dependence.
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

    return knaca*(1./(KmNai*KmNai*KmNai+nao*nao*nao))*(1./(KmCa+cao)) *\
            (1./(1+ksat*exp((n_-1)*u*F/(R*T)))) *\
            (exp(n_*u*F/(R*T))*nai*nai*nai*cao -
                exp((n_-1)*u*F/(R*T))*nao*nao*nao*cai*2.5)

def calc_inak(u, nai, ko, KmK, KmNa, knak, F, R, T, exp=math.exp):
    """
    Calculates the sodium-potassium pump current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    nai : np.ndarray
        Sodium ion concentration in the intracellular space.
    ko : float
        Potassium ion concentration in the extracellular space.
    KmK : float
        Michaelis constant for potassium.
    KmNa : float
        Michaelis constant for sodium.
    knak : float
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

    rec_iNaK = (
        1./(1.+0.1245*exp(-0.1*u*F/(R*T))+0.0353*exp(-u*F/(R*T))))

    return knak*(ko/(ko+KmK))*(nai/(nai+KmNa))*rec_iNaK

def calc_ipca(cai, KpCa, gpca):
    """
    Calculates the calcium pump current.

    Parameters
    ----------
    cai : np.ndarray
        Calcium concentration in the submembrane space.
    KpCa : float
        Michaelis constant for calcium pump.
    gpca : float
        Calcium pump conductance.

    Returns
    -------
    np.ndarray
        Updated calcium pump current array.
    """

    return gpca*cai/(KpCa+cai)

def calc_ipk(u, Ek, gpk, exp=math.exp):
    """
    Calculates the potassium pump current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ek : float
        Potassium reversal potential.
    gpk : float
        Potassium pump conductance.
    
    Returns
    -------
    np.ndarray
        Updated potassium pump current array.
    """
    rec_ipK = 1./(1.+exp((25-u)/5.98))

    return gpk*rec_ipK*(u-Ek)

def calc_ibna(u, Ena, gbna):
    """
    Calculates the background sodium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Ena : float
        Sodium reversal potential.
    gbna : float
        Background sodium conductance.

    Returns
    -------
    np.ndarray
        Updated background sodium current array.
    """

    return gbna*(u-Ena)

def calc_ibca(u, Eca, gbca):
    """
    Calculates the background calcium current.

    Parameters
    ----------
    u : np.ndarray
        Membrane potential array.
    Eca : float
        Calcium reversal potential.
    gbca : float
        Background calcium conductance.

    Returns
    -------
    np.ndarray
        Updated background calcium current array.
    """

    return gbca*(u-Eca)

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

    return maxsr-((maxsr-minsr)/(1+(EC/casr)*(EC/casr)))

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

    return vrel*oo*(casr-cass)

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

    return vleak*(casr-cai)

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

    return vmaxup/(1.+((Kup*Kup)/(cai*cai)))

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

    return vxfer*(cass-cai)

def calc_casr(dt, caSR, bufsr, Kbufsr, iup, irel, ileak, sqrt=math.sqrt):
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

    CaCSQN = bufsr*caSR/(caSR+Kbufsr)
    dCaSR = dt*(iup-irel-ileak)
    bjsr = bufsr-CaCSQN-dCaSR-caSR+Kbufsr
    cjsr = Kbufsr*(CaCSQN+dCaSR+caSR)
    return (sqrt(bjsr*bjsr+4*cjsr)-bjsr)/2

def calc_cass(dt, caSS, bufss, Kbufss, ixfer, irel, ical, capacitance, Vc, Vss, Vsr, inversevssF2, sqrt=math.sqrt):
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
    capacitance : float
        Membrane capacitance.
    Vc : float
        Volume of the cytosol.
    Vss : float
        Volume of the submembrane space.
    Vsr : float
        Volume of the sarcoplasmic reticulum.
    inversevssF2 : float
        Inverse of the product of 2
        times the volume of the submembrane space and Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated calcium concentration in the submembrane space.
    """

    CaSSBuf = bufss*caSS/(caSS+Kbufss)
    dCaSS = dt*(-ixfer*(Vc/Vss)+irel*(Vsr/Vss) +
                (-ical*inversevssF2*capacitance))
    bcss = bufss-CaSSBuf-dCaSS-caSS+Kbufss
    ccss = Kbufss*(CaSSBuf+dCaSS+caSS)
    return (sqrt(bcss*bcss+4*ccss)-bcss)/2

def calc_cai(dt, cai, bufc, Kbufc, ibca, ipca, inaca, iup, ileak, ixfer, capacitance, vsr, vc, inverseVcF2, sqrt=math.sqrt):
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
    capacitance : float
        Membrane capacitance.
    vsr : float
        Volume of the sarcoplasmic reticulum.
    vc : float
        Volume of the cytosol.
    inverseVcF2 : float
        Inverse of the product of 2
        times the volume of the cytosol and Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated calcium concentration in the cytosol.
    """

    CaCBuf = bufc*cai/(cai+Kbufc)
    dCai = dt*((-(ibca+ipca-2*inaca)*inverseVcF2*capacitance) -
                   (iup-ileak)*(vsr/vc)+ixfer)
    bc = bufc-CaCBuf-dCai-cai+Kbufc
    cc = Kbufc*(CaCBuf+dCai+cai)
    return (sqrt(bc*bc+4*cc)-bc)/2

def calc_dnai(ina, ibna, inak, inaca, capacitance, inverseVcF):
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
    capacitance : float
        Membrane capacitance.
    inverseVcF : float
        Inverse of the product of the volume of the cytosol and Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated sodium concentration in the cytosol.
    """

    dNai = -(ina+ibna+3*inak+3*inaca)*inverseVcF*capacitance
    return dNai

def calc_dki(ik1, ito, ikr, iks, inak, ipk, inverseVcF, capacitance):
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
    capacitance : float
        Membrane capacitance.
    inverseVcF : float
        Inverse of the product of the volume of the cytosol and Faraday's constant.

    Returns
    -------
    np.ndarray
        Updated potassium concentration in the cytosol.
    """

    dKi = -(ik1+ito+ikr+iks-2*inak+ipk)*inverseVcF*capacitance
    return dKi
