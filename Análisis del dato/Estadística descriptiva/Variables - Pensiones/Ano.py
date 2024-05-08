"""ESTUDIO DESCRIPTIVO DE LA VARIBALE NUMÉRICA: Año"""
import pandas as pd
import matplotlib.pyplot as plt

# Lee los datos del archivo Excel
df = pd.read_excel("C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/data/altas&bajas_pensiones_sin2023.xlsx")

# Calcula las estadísticas descriptivas para cada columna
ano = df['Año']
estadisticas_ano = ano.describe()
print(estadisticas_ano)

# Gráfico combinado
fig, ax = plt.subplots(figsize=(10, 6))

# Boxplot con color azul claro
bp = ax.boxplot(df['Año'].values, patch_artist=True, boxprops=dict(facecolor='lightblue'))

# Cambiar el grosor de la línea que representa la media
for median in bp['medians']:
    median.set(color='red', linewidth=2)

# Anotaciones
ax.text(1, estadisticas_ano['mean'] + 0.5, f'Media: {estadisticas_ano["mean"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_ano['std'], f'Desv. Estándar: {estadisticas_ano["std"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_ano['min'] - 0.5, f'Mínimo: {estadisticas_ano["min"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_ano['max'] + 0.5, f'Máximo: {estadisticas_ano["max"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_ano['count'], f'Núm. Obs.: {estadisticas_ano["count"]}', va='center', ha='left')

plt.title('Diagrama de caja variable Año')
plt.ylabel('Año')

# Configurar los ticks del eje x
ax.set_xticks([])

# Ajuste manual de los márgenes
plt.subplots_adjust(bottom=0.15, top=0.9, right=0.95)
plt.show()
