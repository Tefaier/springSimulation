import math

import pandas as pd

def calculateNaturalFrequency(k: float, mass: float) -> float:
    return math.sqrt(k / mass)

def calculateNaturalPeriod(k: float, mass: float) -> float:
    return 2 * math.pi * math.sqrt(mass / k)

def calculateHarmonicOscillation(time: pd.Timedelta, frequency: float, amplitude: float) -> float:
    return amplitude * math.sin(frequency * time.total_seconds())

def calculateHarmonicOscillationVelocity(time: pd.Timedelta, frequency: float, amplitude: float) -> float:
    return amplitude * frequency * math.cos(frequency * time.total_seconds())

def calculateHarmonicOscillationAcceleration(time: pd.Timedelta, frequency: float, amplitude: float) -> float:
    return amplitude * frequency ** 2 * math.cos(frequency * time.total_seconds() + math.pi / 2)

# viscosity - вязкость в Па*с
# mass - масса шарика в кг
# density - плотность материала шарика, кг/м^3, по умолчанию железо
def getFrictionCoefficientMas(viscosity: float, mass: float, density: float = 7874) -> float:
    return getFrictionCoefficientRad(viscosity, calculateSphereRadius(mass, density))

def getFrictionCoefficientRad(viscosity: float, radius: float) -> float:
    return 6 * math.pi * radius * viscosity

def getViscosityBackMas(coefficient: float, mass: float, density: float = 7874) -> float:
    return getViscosityBackRad(coefficient, calculateSphereRadius(mass, density))

def getViscosityBackRad(coefficient: float, radius: float) -> float:
    return coefficient / (6 * math.pi * radius)

def calculateSphereRadius(mass: float, density: float = 7874) -> float:
    return ((((mass / density) * 3) / 4) / math.pi) ** (1./3)
