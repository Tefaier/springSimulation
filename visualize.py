import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Simulation.Container import SimulationContainer, calculateNaturalFrequency

sys.setrecursionlimit(3000)

offset_0 = 0.1
simulation = SimulationContainer([100], 1, 100, offset_0, pd.Timedelta(milliseconds=10))
simulation.setObservedSite((1,))
simulation.setForcedOscillation((-1,), 0.03, calculateNaturalFrequency(100 * 2, 1))

simulation.iterate()
data_to_draw = simulation.information[1:-1, 3:4].transpose() - [offset_0]


def animate(i):
    global data_to_draw
    for _ in range(5):
        simulation.iterate()
    data_to_draw = simulation.information[1:-1, 3:4].transpose() - [offset_0]

    ax.cla()
    sns.heatmap(ax=ax, data=data_to_draw, cmap="vlag", cbar_ax=cbar_ax, vmax=0.002, vmin=-0.002)


grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(14, 4))
ani = FuncAnimation(fig=fig, func=animate, frames=100, interval=1)
plt.show()
