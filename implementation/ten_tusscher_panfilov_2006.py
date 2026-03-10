"""
This module provides a simple interface to run the model in a 0D setting,
i.e., without spatial dimensions. It includes class for defining stimulation protocols
and a class for the 0D model itself.

"""

import math

from ten_tusscher_panfilov_2006 import ops


class Stimulation:
    """
    Stimulus protocol for the 0D model.

    Parameters
    ----------
    t_start : float
        Start time (ms) of the first stimulus window.
    duration : float
        Duration (ms) of a single pulse.
    amplitude : float
        Pulse amplitude in the same units as du/dt contribution (typically "units/ms").

    Method
    ------
    stim(t: float) -> float
        Returns the instantaneous stimulus value at time t.

    """

    def __init__(self, t_start: float, duration: float, amplitude: float):
        self.t_start = t_start
        self.duration = duration
        self.amplitude = amplitude

    def stim(self, t: float) -> float:
        return self.amplitude if self.t_start <= t < self.t_start + self.duration else 0.0


class tenTusscherPanfilov20060D:
    """
    ten Tusscher-Panfilov 2006 OD implementation.

    Parameters
    ----------

    dt : float
        Time step size (ms).
    stimulations : list[Stimulation]
        List of stimulation protocols to apply during the simulation.

    Attributes
    ----------
    variables : dict[str, float]
        Current state variables of the model.
    parameters : dict[str, float]
        Model parameters.
    history : dict[str, list[float]]
        Time history of state variables for post-processing.
    
    Methods
    -------
    step(i: int)
        Perform a single time step update.
    run(t_max: float)
        Run the simulation up to time t_max.
    """
    def __init__(self, dt: float, stimulations: list[Stimulation]):
        self.dt = dt
        self.stimulations = stimulations
        self.variables = ops.get_variables()
        self.parameters = ops.get_parameters()
        self.history = {s: [] for s in self.variables}

    def step(self, i: int):
        """
        Perform a single time step update.

        Parameters
        ----------
        i : int
            Current time step index.
        """
        inverseVcF2 = 1.0 / (2 * self.parameters["Vc"] * self.parameters["F"])
        inverseVcF = 1.0 / (self.parameters["Vc"] * self.parameters["F"])
        inversevssF2 = 1.0 / (2 * self.parameters["Vss"] * self.parameters["F"])

        # Old state
        u_old = self.variables["u"]
        m_old = self.variables["m"]
        h_old = self.variables["h"]
        j_old = self.variables["j"]
        d_old = self.variables["d"]
        f_old = self.variables["f"]
        f2_old = self.variables["f2"]
        fcass_old = self.variables["fcass"]
        r_old = self.variables["r"]
        s_old = self.variables["s"]
        xr1_old = self.variables["xr1"]
        xr2_old = self.variables["xr2"]
        xs_old = self.variables["xs"]
        rr_old = self.variables["rr"]
        oo_old = self.variables["oo"]

        casr_old = self.variables["casr"]
        cass_old = self.variables["cass"]
        cai_old = self.variables["cai"]
        nai_old = self.variables["nai"]
        ki_old = self.variables["Ki"]

        # Reversal potentials from old state
        Ek = self.parameters["RTONF"] * math.log(self.parameters["ko"] / ki_old)
        Ena = self.parameters["RTONF"] * math.log(self.parameters["nao"] / nai_old)
        Eks = self.parameters["RTONF"] * math.log(
            (self.parameters["ko"] + self.parameters["pKNa"] * self.parameters["nao"])
            / (ki_old + self.parameters["pKNa"] * nai_old)
        )
        Eca = 0.5 * self.parameters["RTONF"] * math.log(self.parameters["cao"] / cai_old)

        # Gating/state updates from old state
        m_new = ops.calc_gating_m(m_old, u_old, self.dt)
        h_new, h_inf = ops.calc_gating_h(h_old, u_old, self.dt)
        j_new = ops.calc_gating_j(j_old, h_inf, u_old, self.dt)
        
        # calculating currents from old state and new gating/state variables
        ina = ops.calc_ina(
            u_old, m_old, h_old, j_old,
            self.parameters["gna"], Ena
        )

        ical, d_new, f_new, f2_new, fcass_new = ops.calc_ical(
            u_old, self.dt, d_old, f_old, f2_old,
            fcass_old, self.parameters["cao"], cass_old,
            self.parameters["gcal"], self.parameters["F"],
            self.parameters["R"], self.parameters["T"]
        )

        ito, r_new, s_new = ops.calc_ito(
            u_old, self.dt, r_old, s_old, Ek,
            self.parameters["gto"]
        )

        ikr, xr1_new, xr2_new = ops.calc_ikr(
            u_old, self.dt, xr1_old, xr2_old, Ek,
            self.parameters["gkr"], self.parameters["ko"]
        )

        iks, xs_new = ops.calc_iks(
            u_old, self.dt, xs_old, Eks,
            self.parameters["gks"]
        )

        ik1 = ops.calc_ik1(u_old, Ek, self.parameters["gk1"])

        inaca = ops.calc_inaca(
            u_old, self.parameters["nao"], nai_old,
            self.parameters["cao"], cai_old,
            self.parameters["KmNai"], self.parameters["KmCa"],
            self.parameters["knaca"], self.parameters["ksat"],
            self.parameters["n_"], self.parameters["F"],
            self.parameters["R"], self.parameters["T"]
        )

        inak = ops.calc_inak(
            u_old, nai_old, self.parameters["ko"],
            self.parameters["KmK"], self.parameters["KmNa"],
            self.parameters["knak"], self.parameters["F"],
            self.parameters["R"], self.parameters["T"]
        )

        ipca = ops.calc_ipca(
            cai_old, self.parameters["KpCa"],
            self.parameters["gpca"]
        )

        ipk = ops.calc_ipk(
            u_old, Ek,
            self.parameters["gpk"]
        )

        ibna = ops.calc_ibna(
            u_old, Ena,
            self.parameters["gbna"]
        )

        ibca = ops.calc_ibca(
            u_old, Eca,
            self.parameters["gbca"]
        )

        irel, rr_new, oo_new = ops.calc_irel(
            self.dt, rr_old, oo_old,
            casr_old, cass_old,
            self.parameters["Vrel"], self.parameters["k1"],
            self.parameters["k2"], self.parameters["k3"],
            self.parameters["k4"], self.parameters["maxsr"],
            self.parameters["minsr"], self.parameters["EC"]
        )

        ileak = ops.calc_ileak(
            casr_old, cai_old,
            self.parameters["Vleak"]
        )

        iup = ops.calc_iup(
            cai_old,
            self.parameters["Vmaxup"], self.parameters["Kup"]
        )

        ixfer = ops.calc_ixfer(
            cass_old, cai_old,
            self.parameters["Vxfer"]
        )

        # Concentration updates from old state and old-state currents
        casr_new = ops.calc_casr(
            self.dt, casr_old,
            self.parameters["Bufsr"], self.parameters["Kbufsr"],
            iup, irel, ileak
        )

        cass_new = ops.calc_cass(
            self.dt, cass_old,
            self.parameters["Bufss"], self.parameters["Kbufss"],
            ixfer, irel, ical,
            self.parameters["CAPACITANCE"],
            self.parameters["Vc"], self.parameters["Vss"],
            self.parameters["Vsr"], inversevssF2
        )

        cai_new = ops.calc_cai(
            self.dt, cai_old,
            self.parameters["Bufc"], self.parameters["Kbufc"],
            ibca, ipca, inaca, iup, ileak, ixfer,
            self.parameters["CAPACITANCE"],
            self.parameters["Vsr"], self.parameters["Vc"],
            inverseVcF2
        )

        dnai = ops.calc_dnai(
            ina, ibna,
            inak, inaca,
            self.parameters["CAPACITANCE"],
            inverseVcF
        )

        dki = ops.calc_dki(
            ik1, ito,
            ikr, iks,
            inak, ipk,
            inverseVcF, self.parameters["CAPACITANCE"]
        )

        stim_current = sum(stim.stim(t=self.dt * i) for stim in self.stimulations)

        du = -ops.calc_rhs(
            ikr, iks,
            ik1, ito,
            ina, ibna,
            ical, ibca,
            inak, inaca,
            ipca, ipk
        ) + stim_current

        # Commit new state
        self.variables["m"] = m_new
        self.variables["h"] = h_new
        self.variables["j"] = j_new

        self.variables["d"] = d_new
        self.variables["f"] = f_new
        self.variables["f2"] = f2_new
        self.variables["fcass"] = fcass_new

        self.variables["r"] = r_new
        self.variables["s"] = s_new
        self.variables["xr1"] = xr1_new
        self.variables["xr2"] = xr2_new
        self.variables["xs"] = xs_new

        self.variables["rr"] = rr_new
        self.variables["oo"] = oo_new

        self.variables["casr"] = casr_new
        self.variables["cass"] = cass_new
        self.variables["cai"] = cai_new
        self.variables["nai"] = nai_old + self.dt * dnai
        self.variables["Ki"] = ki_old + self.dt * dki

        self.variables["u"] = u_old + self.dt * du

    def run(self, t_max: float):
        """
        Run the simulation up to time t_max.
        
        Parameters
        ----------
        t_max : float
            Maximum simulation time.
        """
        n_steps = int(round(t_max/self.dt))
        for i in range(n_steps):
            self.step(i)
            for s in self.variables:
                self.history[s].append(self.variables[s])
