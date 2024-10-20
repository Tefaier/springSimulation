import pandas as pd

from Simulation.Container import SimulationContainer

if __name__ == "main":
    simulation = SimulationContainer([100], 1, 100, 0.1, pd.Timedelta(milliseconds=10))