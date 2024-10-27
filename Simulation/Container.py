import math
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import brent

from Simulation.Enums import FieldStatIndex, OutputStatIndex, SideType
from Simulation.SimulationMath import calculateNaturalFrequency, calculateHarmonicOscillation, \
    calculateHarmonicOscillationVelocity


class SimulationContainer:
    infoNumber = 6
    startInformation: np.array
    information: np.array # border values represent walls
    dimensions: tuple[int]
    pureDimensions: tuple[int]
    mass: float
    k: float
    offset: float
    time: pd.Timedelta
    deltaT: pd.Timedelta
    observedSide: tuple[int]
    oscillatingSide: tuple[int] = None
    oscillationFrequency: float = None
    oscillationAmplitude: float = None
    oscillationStart: pd.Timedelta = None
    frictionCoefficient: float
    naturalFrequency: float

    def __init__(self, dimensions: List[int], mass: float, k: float, offset: float, deltaT: pd.Timedelta, frictionCoefficient: float = 0):
        self.pureDimensions = tuple(dimensions)
        self.dimensions = tuple(x + 2 for x in dimensions) + (self.infoNumber,)
        self.information = np.zeros(self.dimensions, dtype=float)
        self.mass = mass
        self.k = k
        self.time = pd.Timedelta(seconds=0)
        self.frictionCoefficient = frictionCoefficient
        self.deltaT = deltaT
        self.offset = offset
        self.setObservedSite((1,))
        self.naturalFrequency = calculateNaturalFrequency(self.k * 2, self.mass)
        posX = np.arange(0, self.dimensions[0], dtype=float) * offset
        # np.repeat + np.reshape when there are more dimensions
        self.information[:, FieldStatIndex.LocationX.value] = posX
        self.information[:-1, FieldStatIndex.OffsetX.value] = self.information[1:, FieldStatIndex.LocationX.value] - self.information[:-1, FieldStatIndex.LocationX.value]
        # copy initial information for use
        self.startInformation = self.information.copy()

    def setObservedSite(self, side: tuple[int]):
        self.observedSide = side

    def stopOscillation(self):
        if self.oscillatingSide == None: return
        proxyArray, _, side = self.getWallProxy(self.oscillatingSide)
        if side == SideType.x:
            proxyArray[FieldStatIndex.LocationX.value] = self.startInformation[FieldStatIndex.LocationX.value]
        self.oscillationStart = None

    def setForcedOscillation(self, side: tuple[int], amplitude: float, frequency: float = None):
        self.stopOscillation()
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
        # do iteration stuff here
        self.information[1:-1, FieldStatIndex.ForceX.value] = (
            (self.offset - self.information[:-2, FieldStatIndex.OffsetX.value]) * self.k +
            (self.information[1:-1, FieldStatIndex.OffsetX.value] - self.offset) * self.k -
            self.information[1:-1, FieldStatIndex.VelocityX.value] * self.frictionCoefficient
        )
        # new pure velocity
        self.information[1:-1, FieldStatIndex.VelocityX.value] += self.information[1:-1, FieldStatIndex.ForceX.value] * self.deltaT.total_seconds() / self.mass
        # trying to apply and as well limit by energy
        self.information[1:-1, FieldStatIndex.Energy.value] = (
            np.power(self.information[1:-1, FieldStatIndex.VelocityX.value], 2) * self.mass / 2 +
            np.power(
                np.maximum(self.information[:-2, FieldStatIndex.OffsetX.value], self.information[1:-1, FieldStatIndex.OffsetX.value]) -
                (self.information[:-2, FieldStatIndex.OffsetX.value] + self.information[1:-1, FieldStatIndex.OffsetX.value]) / 2, 2) *
            self.k / 2
        )
        # simple location recalculation
        self.information[1:-1, FieldStatIndex.LocationX.value] += (
            self.information[1:-1, FieldStatIndex.VelocityX.value] * self.deltaT.total_seconds()
        )
        self.information[:-1, FieldStatIndex.OffsetX.value] = self.information[1:, FieldStatIndex.LocationX.value] - self.information[:-1, FieldStatIndex.LocationX.value]
        # enforce energy preservation
        newEnergy = (
            np.power(self.information[1:-1, FieldStatIndex.VelocityX.value], 2) * self.mass / 2 +
            np.power(
                np.maximum(self.information[:-2, FieldStatIndex.OffsetX.value], self.information[1:-1, FieldStatIndex.OffsetX.value]) -
                (self.information[:-2, FieldStatIndex.OffsetX.value] + self.information[1:-1, FieldStatIndex.OffsetX.value]) / 2, 2) *
            self.k / 2
        )
        # if new energy is higher than should be
        energyHigher = np.where(newEnergy >= self.information[1:-1, FieldStatIndex.Energy.value])
        amplitudeOvershoot = np.where(
            np.power(
                np.maximum(self.information[:-2, FieldStatIndex.OffsetX.value], self.information[1:-1, FieldStatIndex.OffsetX.value]) -
                (self.information[:-2, FieldStatIndex.OffsetX.value] + self.information[1:-1, FieldStatIndex.OffsetX.value]) / 2, 2) *
            self.k / 2 > self.information[1:-1, FieldStatIndex.Energy.value]
        )
        # overshoot in positive direction
        self.information[1:-1, FieldStatIndex.LocationX.value][np.logical_and(amplitudeOvershoot, np.where(self.information[:-2, FieldStatIndex.OffsetX.value] > self.information[1:-1, FieldStatIndex.OffsetX.value]))] = (
            self.information[1:-1, FieldStatIndex.LocationX.value] - (
                self.information[:-2, FieldStatIndex.OffsetX.value] - (
                self.information[:-2, FieldStatIndex.OffsetX.value] + self.information[1:-1,
                                                                      FieldStatIndex.OffsetX.value]) / 2 -
                np.sqrt(self.information[1:-1, FieldStatIndex.Energy.value] * 2 / self.k)
            )
        )
        # overshoot in negative direction
        self.information[1:-1, FieldStatIndex.LocationX.value][np.logical_and(amplitudeOvershoot, np.where(
            self.information[:-2, FieldStatIndex.OffsetX.value] < self.information[1:-1,
                                                                  FieldStatIndex.OffsetX.value]))] = (
                self.information[1:-1, FieldStatIndex.LocationX.value] - (
                self.information[:-2, FieldStatIndex.OffsetX.value] - (
                self.information[:-2, FieldStatIndex.OffsetX.value] + self.information[1:-1,
                                                                      FieldStatIndex.OffsetX.value]) / 2 +
                np.sqrt(self.information[1:-1, FieldStatIndex.Energy.value] * 2 / self.k)
        )
        )
        self.information[1:-1, FieldStatIndex.VelocityX.value][amplitudeOvershoot] = 0
        # only speed overshoot to be solved
        self.information[1: -1, FieldStatIndex.VelocityX.value][np.logical_xor(energyHigher, amplitudeOvershoot)] = (
            np.sqrt((self.information[1:-1, FieldStatIndex.Energy.value] - np.power(
                np.maximum(self.information[:-2, FieldStatIndex.OffsetX.value], self.information[1:-1, FieldStatIndex.OffsetX.value]) -
                (self.information[:-2, FieldStatIndex.OffsetX.value] + self.information[1:-1, FieldStatIndex.OffsetX.value]) / 2, 2) *
            self.k / 2) * 2 / self.mass) * np.sign(self.information[1: -1, FieldStatIndex.VelocityX.value])
        )
        # energy is lower so speed is increased
        energyLower = np.logical_not(energyHigher)
        self.information[1: -1, FieldStatIndex.VelocityX.value][energyLower] = (
                np.sqrt((self.information[1:-1, FieldStatIndex.Energy.value] - np.power(
                    np.maximum(self.information[:-2, FieldStatIndex.OffsetX.value],
                               self.information[1:-1, FieldStatIndex.OffsetX.value]) -
                    (self.information[:-2, FieldStatIndex.OffsetX.value] + self.information[1:-1,
                                                                           FieldStatIndex.OffsetX.value]) / 2, 2) *
                         self.k / 2) * 2 / self.mass) * np.sign(self.information[1: -1, FieldStatIndex.VelocityX.value])
        )
        self.information[:-1, FieldStatIndex.OffsetX.value] = self.information[1:, FieldStatIndex.LocationX.value] - self.information[:-1, FieldStatIndex.LocationX.value]




    def generateReturn(self) -> np.array:
        returnSize = 2
        proxyArray, proxyArrayPrevious, _ = self.getSideProxy(self.observedSide)

        returnArr = np.zeros((returnSize) if len(proxyArray.shape) == 1 else tuple([x - 2 for x in proxyArray.shape]) + (returnSize,), dtype=float)
        # amplitude
        # mv^2/2 + mw^2x^2/2 = mw^2X^2/2
        # v^2/w^2 + x^2 = X^2
        returnArr[OutputStatIndex.Amplitude.value] = (
            np.sqrt(
                np.power(proxyArray[FieldStatIndex.VelocityX.value], 2) / (self.naturalFrequency**2) +
                np.power(np.maximum(proxyArray[FieldStatIndex.OffsetX.value], proxyArrayPrevious[FieldStatIndex.OffsetX.value]) -
                         (proxyArray[FieldStatIndex.OffsetX.value] + proxyArrayPrevious[FieldStatIndex.OffsetX.value]) / 2, 2)
            ))
        # force, in positive index direction
        returnArr[OutputStatIndex.Force.value] = (
            (proxyArray[FieldStatIndex.OffsetX.value] - self.offset) * self.k +
            (self.offset - proxyArrayPrevious[FieldStatIndex.OffsetX.value]) * self.k
        )

        return returnArr

    def performForcedOscilation(self):
        if self.oscillationStart is None: return
        proxyArray, proxyStart, side = self.getWallProxy(self.oscillatingSide)
        offset = calculateHarmonicOscillation(self.time - self.oscillationStart, self.oscillationFrequency, self.oscillationAmplitude)
        velocity = calculateHarmonicOscillationVelocity(self.time - self.oscillationStart, self.oscillationFrequency, self.oscillationAmplitude)
        if side == SideType.x:
            proxyArray[FieldStatIndex.LocationX.value] = proxyStart[FieldStatIndex.LocationX.value] + offset
            proxyArray[FieldStatIndex.VelocityX.value] = velocity

    # [0] is array of ones that are at the required side
    # [1] is array of ones that are just before [0] in terms on index position
    # it is 1D currently
    # anyway the idea is that it is tuple of zeroes except on value being -1 or 1
    def getSideProxy(self, side: tuple[int]) -> tuple[np.array, np.array, SideType]:
        if side[0] == -1:
            return (self.information[1], self.information[0], SideType.x)
        else:
            return (self.information[-2], self.information[-3], SideType.x)

    def getWallProxy(self, side: tuple[int]) -> tuple[np.array, np.array, SideType]:
        if side[0] == -1:
            return (self.information[0], self.startInformation[0], SideType.x)
        else:
            return (self.information[-1], self.startInformation[-1], SideType.x)
