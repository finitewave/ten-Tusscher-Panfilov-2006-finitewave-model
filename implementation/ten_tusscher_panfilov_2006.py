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
        inverseVcF2 = 1./(2*self.parameters["Vc"]*self.parameters["F"]) 
        inverseVcF = 1./(self.parameters["Vc"]*self.parameters["F"])
        inversevssF2 = 1./(2*self.parameters["Vss"]*self.parameters["F"])

        Ek = self.parameters["RTONF"]*(math.log((self.parameters["ko"]/self.variables["Ki"])))
        Ena = self.parameters["RTONF"]*(math.log((self.parameters["nao"]/self.variables["nai"])))
        
        Eks = self.parameters["RTONF"]*(math.log((self.parameters["ko"]+self.parameters["pkNa"]*self.parameters["nao"])/(self.variables["Ki"]+self.parameters["pkNa"]*self.variables["nai"])))
        Eca = 0.5*self.parameters["RTONF"]*(math.log((self.parameters["cao"]/self.variables["cai"])))

        # Compute currents
        ina, self.variables["m"], self.variables["h"], self.variables["j_"] = ops.calc_ina(
            self.variables["u"], self.dt, self.variables["m"], self.variables["h"], self.variables["j_"],
            self.parameters["gna"], Ena)
        ical, self.variables["d"], self.variables["f"], self.variables["f2"], self.variables["fcass"] = ops.calc_ical(
            self.variables["u"], self.dt, self.variables["d"], self.variables["f"], self.variables["f2"],
            self.variables["fcass"], self.parameters["cao"], self.variables["cass"],
            self.parameters["gcal"], self.parameters["F"], self.parameters["R"], self.parameters["T"])
        ito, self.variables["r"], self.variables["s"] = ops.calc_ito(
            self.variables["u"], self.dt, self.variables["r"], self.variables["s"], Ek,
            self.parameters["gto"])
        ikr, self.variables["xr1"], self.variables["xr2"] = ops.calc_ikr(
            self.variables["u"], self.dt, self.variables["xr1"], self.variables["xr2"], Ek,
            self.parameters["gkr"], self.parameters["ko"])
        iks, self.variables["xs"] = ops.calc_iks(
            self.variables["u"], self.dt, self.variables["xs"], Eks,
            self.parameters["gks"])
        ik1 = ops.calc_ik1(self.variables["u"], Ek, self.parameters["gk1"])
        inaca = ops.calc_inaca(
            self.variables["u"], self.parameters["nao"], self.variables["nai"],
            self.parameters["cao"], self.variables["cai"],
            self.parameters["KmNai"], self.parameters["KmCa"],
            self.parameters["knaca"], self.parameters["ksat"],
            self.parameters["n_"], self.parameters["F"],
            self.parameters["R"], self.parameters["T"])
        inak = ops.calc_inak(
            self.variables["nai"], self.parameters["ko"],
            self.parameters["KmK"], self.parameters["KmNa"],
            self.parameters["knak"], self.parameters["F"],
            self.parameters["R"], self.parameters["T"])
        ipca = ops.calc_ipca(
            self.variables["cai"], self.parameters["KpCa"],
            self.parameters["gpca"])
        ipk = ops.calc_ipk(
            self.variables["u"], Ek,
            self.parameters["gpk"])
        ibna = ops.calc_ibna(
            self.variables["u"], Ena,
            self.parameters["gbna"])
        ibca = ops.calc_ibca(
            self.variables["u"], Eca,
            self.parameters["gbca"])
        irel, self.variables["rr"], self.variables["oo"] = ops.calc_irel(
            self.dt, self.variables["rr"], self.variables["oo"],
            self.variables["casr"], self.variables["cass"],
            self.parameters["Vrel"], self.parameters["k1_"],
            self.parameters["k2_"], self.parameters["k3"],
            self.parameters["k4"], self.parameters["maxsr"],
            self.parameters["minsr"], self.parameters["EC"])
        ileak = ops.calc_ileak(
            self.variables["casr"], self.variables["cai"],
            self.parameters["Vleak"])
        iup = ops.calc_iup(
            self.variables["cai"],
            self.parameters["Vmaxup"], self.parameters["Kup"])
        ixfer = ops.calc_ixfer(
            self.variables["cass"], self.variables["cai"],
            self.parameters["Vxfer"])
        
        # Compute concentrations
        self.variables["casr"] = ops.calc_casr(
            self.dt, self.variables["casr"],
            self.parameters["Bufsr"], self.parameters["Kbufsr"],
            iup, irel, ileak)
        self.variables["cass"] = ops.calc_cass(
            self.dt, self.variables["cass"],
            self.parameters["Bufss"], self.parameters["Kbufss"],
            ixfer, irel, ical,
            self.parameters["CAPACITANCE"],
            self.parameters["Vc"], self.parameters["Vss"],
            self.parameters["Vsr"], inversevssF2)
        self.variables["cai"], self.variables["cai"] = ops.calc_cai(
            self.dt, self.variables["cai"],
            self.parameters["Bufc"], self.parameters["Kbufc"],
            ibca, ipca, inaca, iup, ileak, ixfer,
            self.parameters["CAPACITANCE"],
            self.parameters["Vsr"], self.parameters["Vc"],
            inverseVcF2)
        self.variables["nai"] += self.dt*ops.calc_dnai(
            ina, ibna,
            inak, inaca,
            self.parameters["CAPACITANCE"],
            inverseVcF)
        self.variables["Ki"] += self.dt*ops.calc_dki(
            ik1, ito,
            ikr, iks,
            inak, ipk,
            inverseVcF, self.parameters["CAPACITANCE"])

        # Update membrane potential
        self.variables["u"] += self.dt * (-ops.calc_rhs(
            ikr, iks,
            ik1, ito,
            ina, ibna,
            ical, ibca,
            inak, inaca,
            ipca, ipk))

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