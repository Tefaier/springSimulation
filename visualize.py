import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from Simulation.Container import SimulationContainer, calculateNaturalFrequency

kMaxRecursionLimit = 3000
kMillisecondsBetweenFrames = 20
kSimulationStep = kMillisecondsBetweenFrames * 1
offset_0 = 0.1


sys.setrecursionlimit(kMaxRecursionLimit)
simulation = SimulationContainer([30], 0.1, 10, offset_0, pd.Timedelta(milliseconds=1))
simulation.setObservedSite((1,))
simulation.setForcedOscillation((-1,), 0.03, calculateNaturalFrequency(100 * 2, 1))


max_delta = 0.01
def animate(i):
    global max_delta
    for _ in range(kSimulationStep):
        simulation.iterate()
    data_to_draw = simulation.information[1:-1, 3:4].transpose() - [offset_0]
    max_delta = max(max_delta, data_to_draw.max())
    print(max_delta, round(data_to_draw.sum(), 2))
    # print(data_to_draw)

    ax.cla()
    sns.heatmap(ax=ax, data=data_to_draw, cmap="vlag", cbar_ax=cbar_ax, vmax=max_delta, vmin=-max_delta)


grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(14, 4))
ani = FuncAnimation(fig=fig, func=animate, frames=100, interval=kMillisecondsBetweenFrames)
plt.show()
