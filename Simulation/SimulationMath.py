import math

import pandas as pd


def calculateNaturalFrequency(k: float, mass: float) -> float:
    return math.sqrt(k / mass) * 2 * math.pi

def calculateHarmonicOscillation(time: pd.Timedelta, frequency: float, amplitude: float) -> float:
    return amplitude * math.sin(frequency * time.total_seconds())

def calculateHarmonicOscillationVelocity(time: pd.Timedelta, frequency: float, amplitude: float) -> float:
    return frequency * amplitude * math.cos(frequency * time.total_seconds())