import math

import pandas as pd

def calculateNaturalFrequency(k: float, mass: float) -> float:
    return math.sqrt(k / mass) * 2 * math.pi

def calculateHarmonicOscillation(time: pd.Timedelta, frequency: float, amplitude: float) -> float:
    return amplitude * math.sin(frequency * time.total_seconds())

def calculateHarmonicOscillationSpeed(time: pd.Timedelta, frequency: float, amplitude: float) -> float:
    return amplitude * frequency * math.cos(frequency * time.total_seconds())

def calculateHarmonicOscillationVelocity(time: pd.Timedelta, frequency: float, amplitude: float) -> float:
    return amplitude * frequency ** 2 * math.cos(frequency * time.total_seconds() + math.pi / 2)
