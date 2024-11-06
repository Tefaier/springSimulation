import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulation.Container import SimulationContainer, OscillationInfo
from Simulation.Enums import FieldStatIndex
from Simulation.SimulationMath import calculateNaturalFrequency, calculateNaturalPeriod
from Simulation import Enums

timeToIterate = pd.Timedelta(minutes=0.1)
ratiosCheck = [0.001, 0.003, 0.005, 0.01]
offsetToAmpl = [1, 3, 0.4, 0.1, 0.02]
kMassCheck = [
    [0.1, 100],
    [0.2, 200],
    [0.4, 400],
    [0.1, 10],
    [0.1, 200],
    [0.5, 100]
]
freqCheck = [1, 0.1, 0.5, 1.5, 3, 5, 10]
stopAfter = [pd.Timedelta(minutes=10), pd.Timedelta(seconds=1)]
frictionCheck = [0, 0.001, 0.01, 0.05]
offset = 0.5
# check at different ratios
# check at different ratios mass to k
# check at not natural frequency

# fric = frictionCheck[0] # checked
# st = stopAfter[0] # checked
# fr = freqCheck[0] # checked
# off = offsetToAmpl[2] # checked
# kMass = kMassCheck[3] # checked
# r = ratiosCheck[0] # checked
#
# oscAmpl = offset * off
# k = kMass[1]
# mass = kMass[0]
# oscillationAngSpeed = calculateNaturalFrequency(mass=mass, k=k) * fr * 2
# referencePeriod = pd.Timedelta(seconds = (2 * math.pi / calculateNaturalFrequency(mass=mass, k=k)))
# simulation = SimulationContainer([1], mass, k, offset, referencePeriod * r, fric)
# simulation.setForcedOscillation([
#     OscillationInfo((-1,), oscAmpl, oscillationAngSpeed),
# ])

# kMillisecondsBetweenFrames = 20
# kSimulationStep = kMillisecondsBetweenFrames
offset_0 = 0.1
mass = 0.1
k = 100
amount_of_weights = 41

simulation = SimulationContainer([amount_of_weights], mass, k, offset_0, pd.Timedelta(milliseconds=0.1), 0)
simulation.setForcedOscillation([
    OscillationInfo((-1,), offset_0 / 4, calculateNaturalFrequency(mass=mass, k=k), 1),
    # OscillationInfo((1,), offset_0 / 4, -calculateNaturalFrequency(mass=mass, k=k), -1)
])


output = []
time = []
stopped = False
while True:
    simulation.iterate()
    if not stopped and simulation.time > pd.Timedelta(minutes=0.2):
        simulation.stopOscillation()
        stopped = True
    if simulation.time > timeToIterate:
        break
    output.append(simulation.information[1: -1, FieldStatIndex.LocationX.value].copy())
    time.append(simulation.time.total_seconds())
output = np.array(output)
fig = plt.figure(figsize=(10, 5))
for i in range(0, len(output[0])):
    plt.plot(time, output[:, i])
plt.xlabel('t, с')
plt.ylabel('x, m')
# plt.plot([], [], ' ', label="k=" + str(k))
# plt.plot([], [], ' ', label="m=" + str(mass))
# plt.plot([], [], ' ', label="fr=" + str(fr) + " natural")
# plt.plot([], [], ' ', label="amp=" + str(off) + " offset")
# plt.plot([], [], ' ', label=str(round(1 / r, 1)) + " per period")
# plt.plot([], [], ' ', label="fric=" + str(fric))
# plt.legend(loc="upper left")
plt.title('График зависимости позиции от времени')
# plt.show()
plt.savefig("amount_of_weights=" + str(amount_of_weights) + ".png", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
