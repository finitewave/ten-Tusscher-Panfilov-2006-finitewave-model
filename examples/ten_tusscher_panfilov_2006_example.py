"""
Example script to run a 0D model simulation and plot the results.

This script sets up a simple stimulation protocol, runs the simulation,
and plots the membrane potential over time.
"""

import numpy as np
import matplotlib.pyplot as plt

from implementation.ten_tusscher_panfilov_2006 import tenTusscherPanfilov20060D, Stimulation


stimulations = [Stimulation(t_start=100, duration=1., amplitude=30.0)]
t_max = 600.0

model = tenTusscherPanfilov20060D(dt=0.01, stimulations=stimulations)
model.run(t_max=t_max)


# fig = plt.figure()
plt.plot(model.times, model.history['u'], lw=2)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (u)')
plt.title('0D Model Simulation')
plt.grid()
plt.show()

# fig.savefig("ten_tusscher_panfilov_2006_ap.png", dpi=300)
