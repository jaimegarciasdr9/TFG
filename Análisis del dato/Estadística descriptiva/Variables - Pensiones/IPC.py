import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Lee los datos del archivo Excel
df = pd.read_excel("C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/data/altas&bajas_pensiones_sin2023.xlsx")

# Calcula las estadísticas descriptivas para cada columna
ipc = df['IPC']
estadisticas_ipc = ipc.describe()
print(estadisticas_ipc)

# Gráfico combinado - Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(df['IPC'].values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
for median in bp['medians']:
    median.set(color='red', linewidth=2)
ax.text(1, estadisticas_ipc['mean'] + 0.5, f'Media: {estadisticas_ipc["mean"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_ipc['min'] - 0.5, f'Mínimo: {estadisticas_ipc["min"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_ipc['max'] + 0.5, f'Máximo: {estadisticas_ipc["max"]:.2f}', va='center', ha='left')
ax.set_xticks([])
plt.subplots_adjust(bottom=0.15, top=0.9, right=0.95)
plt.title('Boxplot de la variable IPC')
plt.show()

# Gráfico de relación entre Año y Tasa de Desempleo con Seaborn
plt.figure(figsize=(8, 8))
sns.regplot(x="Año", y="IPC", data=df, scatter_kws={"color": "red"}, line_kws={"color": "black"})
media_pib = df['IPC'].mean()
for index, row in df.iterrows():
    color = 'blue' if row['IPC'] > media_pib else 'red'
    plt.scatter(row['Año'], row['IPC'], color=color)
for index, row in df.iterrows():
    plt.text(row['Año'], row['IPC']+0.5, f"{row['IPC']:.2f}", ha='center', va='bottom')
plt.xticks(rotation=25)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
plt.title('Relación entre el Año y el IPC')
plt.xlabel('Año')
plt.ylabel('IPC')
plt.show()