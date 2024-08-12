import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

heights = np.random.uniform(1.4, 2.0, 100)
weights = []

for height in heights:
    min_weight = 18.5 * (height ** 2)
    max_weight = 24.9 * (height ** 2)
    weight = np.random.uniform(min_weight, max_weight)
    weights.append(weight)

data = pd.DataFrame({
    'Estatura (m)': heights,
    'Peso (kg)': weights
})

x = data['Estatura (m)']
y = data['Peso (kg)']

m = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
b = np.mean(y) - m * np.mean(x)

y_line = m * x + b

plt.scatter(data['Estatura (m)'], data['Peso (kg)'], color='blue', label='Datos')
plt.plot(x, y_line, color='red', label='Línea ajustada')
plt.title('Estatura vs Peso con Línea Ajustada')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.show()