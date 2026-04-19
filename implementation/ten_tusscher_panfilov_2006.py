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
        self.stim_history = []
        self.times = []

    def step(self, i: int):
        """
        Perform a single time step update.

        Parameters
        ----------
        i : int
            Current time step index.
        """
        res = ops.ionic_step(self.dt, **self.variables, **self.parameters)
        (rhs, cai_new, casr_new, cass_new, nai_new, ki_new, m_new, h_new, j_new,
         xr1_new, xr2_new, xs_new, r_new, s_new, d_new, f_new, f2_new, fcass_new, rr_new) = res

        
        stim_curr = self.dt * sum(stim.stim(t=self.dt*i) for stim in self.stimulations)
        self.stim_history.append(stim_curr)

        self.variables["u"] += self.dt * rhs + stim_curr  

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

        self.variables["casr"] = casr_new
        self.variables["cass"] = cass_new
        self.variables["cai"] = cai_new
        self.variables["nai"] = nai_new
        self.variables["ki"] = ki_new

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
            self.times.append(self.dt * i)
            for s in self.variables:
                self.history[s].append(self.variables[s])
