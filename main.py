from typing import List

import pandas as pd
import numpy as np

from Simulation.Container import SimulationContainer

if __name__ == "main":
    simulation = SimulationContainer([100], 1, 100, 0.1, pd.Timedelta(milliseconds=10))
    output = list()
    for i in range(0, 100):
        output.append(simulation.iterate())