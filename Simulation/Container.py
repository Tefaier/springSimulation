import math
from typing import List

import numpy as np
import pandas as pd

from Simulation.Enums import FieldStatIndex, OutputStatIndex


class SimulationContainer:
    infoNumber = 4
    startInformation: np.array
    information: np.array # border values represent walls
    dimensions: tuple[int]
    mass: float
    k: float
    offset: float
    time: pd.Timedelta
    deltaT: pd.Timedelta
    observedSide: tuple[int]

    def __init__(self, dimensions: List[int], mass: float, k: float, offset: float, deltaT: pd.Timedelta):
        self.dimensions = tuple(x + 2 for x in dimensions) + tuple(self.infoNumber,)
        self.information = np.zeros(self.dimensions, dtype=float)
        self.mass = mass
        self.k = k
        self.time = pd.Timedelta(seconds=0)
        self.deltaT = deltaT
        self.offset = offset
        self.setObservedSite((True))
        posX = np.arange(0, self.dimensions[0], dtype=float) * offset
        # np.repeat + np.reshape when there are more dimensions
        self.information[:, FieldStatIndex.LocationX] = posX
        self.startInformation = self.information.copy()

    def setObservedSite(self, side: tuple[int]):
        self.observedSide = side

    def iterate(self) -> np.array:
        self.time += self.deltaT
        self.simpleIteration()
        return self.generateReturn()

    def simpleIteration(self):
        self.information[:, :-1, FieldStatIndex.OffsetX] = self.information[:, 1:, FieldStatIndex.LocationX] - self.information[:, :-1, FieldStatIndex.LocationX]
        # do iteration stuff here

    def generateReturn(self) -> np.array:
        returnSize = 2
        naturalFrequency = math.sqrt(2 * self.k / self.mass)
        proxyArray: np.array
        proxyArrayPrevious: np.array
        if self.observedSide[0] == -1:
            proxyArray: np.array = self.information[1]
            proxyArrayPrevious: np.array = self.information[0]
        else:
            proxyArray: np.array = self.information[-2]
            proxyArrayPrevious: np.array = self.information[-3]

        returnArr = np.zeros((returnSize) if len(proxyArray.shape) == 1 else tuple([x - 2 for x in proxyArray.shape]) + tuple(returnSize,), dtype=float)
        # amplitude
        # mv^2/2 + mw^2x^2/2 = mw^2X^2/2
        # v^2/w^2 + x^2 = X^2
        returnArr[OutputStatIndex.Amplitude] = (
            np.sqrt(
                np.power(proxyArray[FieldStatIndex.VelocityX], 2) / (naturalFrequency**2) +
                np.power(np.maximum(proxyArray[FieldStatIndex.LocationX], proxyArrayPrevious[FieldStatIndex.LocationX]), 2)
            ))
        # force, in positive index direction
        returnArr[OutputStatIndex.Force] = (
            (proxyArray[FieldStatIndex.OffsetX] - self.offset) * self.k +
            (self.offset - proxyArrayPrevious[FieldStatIndex.OffsetX]) * self.k
        )

        return returnArr



