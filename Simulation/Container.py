import math
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import brent

from Simulation.Enums import FieldStatIndex, OutputStatIndex, SideType
from Simulation.SimulationMath import calculateNaturalFrequency, calculateHarmonicOscillation, \
    calculateHarmonicOscillationVelocity


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
    oscillatingSide: tuple[int]
    oscillationFrequency: float = None
    oscillationAmplitude: float = None
    oscillationStart: pd.Timedelta

    def __init__(self, dimensions: List[int], mass: float, k: float, offset: float, deltaT: pd.Timedelta):
        self.dimensions = tuple(x + 2 for x in dimensions) + tuple(self.infoNumber,)
        self.information = np.zeros(self.dimensions, dtype=float)
        self.mass = mass
        self.k = k
        self.time = pd.Timedelta(seconds=0)
        self.deltaT = deltaT
        self.offset = offset
        self.setObservedSite((1,))
        posX = np.arange(0, self.dimensions[0], dtype=float) * offset
        # np.repeat + np.reshape when there are more dimensions
        self.information[:, FieldStatIndex.LocationX] = posX
        # copy initial information for use
        self.startInformation = self.information.copy()

    def setObservedSite(self, side: tuple[int]):
        self.observedSide = side

    def stopOscillation(self):
        self.oscillationStart = None

    def setForcedOscillation(self, side: tuple[int], amplitude: float, frequency: float = None):
        self.oscillationStart = self.time
        self.oscillatingSide = side
        self.oscillationAmplitude = amplitude
        self.oscillationFrequency = calculateNaturalFrequency(self.k * 2, self.mass) if frequency is None else frequency

    def iterate(self) -> np.array:
        self.time += self.deltaT
        self.performForcedOscilation()
        self.simpleIteration()
        return self.generateReturn()

    def simpleIteration(self):
        self.information[:, :-1, FieldStatIndex.OffsetX] = self.information[:, 1:, FieldStatIndex.LocationX] - self.information[:, :-1, FieldStatIndex.LocationX]
        # do iteration stuff here

    def generateReturn(self) -> np.array:
        returnSize = 2
        naturalFrequency = calculateNaturalFrequency(self.k * 2, self.mass)
        proxyArray, proxyArrayPrevious, _ = self.getSideProxy(self.observedSide)

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

    def performForcedOscilation(self):
        if self.oscillationStart is None: return
        proxyArray, side = self.getWallProxy(self.oscillatingSide)
        offset = calculateHarmonicOscillation(self.time - self.oscillationStart, self.oscillationFrequency, self.oscillationAmplitude)
        velocity = calculateHarmonicOscillationVelocity(self.time - self.oscillationStart, self.oscillationFrequency, self.oscillationAmplitude)
        if side == SideType.x:
            proxyArray[FieldStatIndex.LocationX] = self.startInformation[FieldStatIndex.LocationX] + offset
            proxyArray[FieldStatIndex.VelocityX] = velocity

    # [0] is array of ones that are at the required side
    # [1] is array of ones that are just before [0] in terms on index position
    # it is 1D currently
    # anyway the idea is that it is tuple of zeroes except on value being -1 or 1
    def getSideProxy(self, side: tuple[int]) -> tuple[np.array, np.array, SideType]:
        if side[0] == -1:
            return (self.information[1], self.information[0], SideType.x)
        else:
            return (self.information[-2], self.information[-3], SideType.x)

    def getWallProxy(self, side: tuple[int]) -> tuple[np.array, SideType]:
        if side[0] == -1:
            return (self.information[0], SideType.x)
        else:
            return (self.information[-1], SideType.x)
