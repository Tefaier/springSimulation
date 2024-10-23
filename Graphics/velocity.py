import pandas as pd
import matplotlib.pyplot as plt
from Simulation.Container import SimulationContainer
from Simulation.SimulationMath import calculateNaturalFrequency
from Simulation import Enums

countIterations = 100
simulation = SimulationContainer([100], 1, 100, 0.1, pd.Timedelta(milliseconds=10))
simulation.setObservedSite((1,))
simulation.setForcedOscillation((-1,), 0.03, calculateNaturalFrequency(100 * 2, 1))
output = []
forces1 = []
forces2 = []
forces3 = []
forces4 = []
forces5 = []
time = []
for i in range(0, countIterations):
    output.append(simulation.iterate())
    forces1.append(simulation.information[1][Enums.FieldStatIndex.VelocityX.value])
    forces2.append(simulation.information[2][Enums.FieldStatIndex.VelocityX.value])
    forces3.append(simulation.information[3][Enums.FieldStatIndex.VelocityX.value])
    forces4.append(simulation.information[4][Enums.FieldStatIndex.VelocityX.value])
    forces5.append(simulation.information[5][Enums.FieldStatIndex.VelocityX.value])
for i in range(countIterations):
    time.append(i * 0.01)
plt.plot(time, forces1, label='Первое тело')
plt.plot(time, forces2, label='Второе тело')
plt.plot(time, forces3, label='Третье тело')
plt.plot(time, forces4, label='Четвёртое тело')
plt.plot(time, forces5, label='Пятое тело')
plt.xlabel('Ось t, с')
plt.ylabel('Ось v, м/с')
plt.title('График зависимости скорости тела от времени')
plt.legend()
plt.show()
