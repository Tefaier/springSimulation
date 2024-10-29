from typing import List

import pandas as pd
import numpy as np

from Simulation.Container import SimulationContainer, OscillationInfo
from Simulation.SimulationMath import calculateNaturalFrequency

if __name__ == "__main__":
    mass = 1
    k = 10
    offset = 0.1
    simulation = SimulationContainer([100], mass, k, offset, pd.Timedelta(milliseconds=10))
    simulation.setForcedOscillation([
        OscillationInfo((-1,), offset / 3, calculateNaturalFrequency(mass=mass, k=k)),
    ])
    mask = np.full(shape=(102,), fill_value=False, dtype=bool)
    mask[50] = True
    simulation.setForcedDisplacement(mask, offset)
    for i in range(0, 10000):
        simulation.iterate()