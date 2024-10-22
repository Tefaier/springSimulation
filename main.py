from typing import List

import pandas as pd
import numpy as np

from Simulation.Container import SimulationContainer
from Simulation.SimulationMath import calculateNaturalFrequency

if __name__ == "main":
    simulation = SimulationContainer([100], 1, 100, 0.1, pd.Timedelta(milliseconds=10))
    simulation.setObservedSite((1,))
    simulation.setForcedOscillation((-1,), 0.03, calculateNaturalFrequency(100 * 2, 1))
    output = list()
    for i in range(0, 100):
        output.append(simulation.iterate())