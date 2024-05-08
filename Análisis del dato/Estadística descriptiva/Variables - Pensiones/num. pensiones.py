import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Lee los datos del archivo Excel
df = pd.read_excel("C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/data/altas&bajas_pensiones_sin2023.xlsx")

# Calcula las estadísticas descriptivas para cada columna
num_pen = df['Núm. pensiones']
estadisticas_num_pen = num_pen.describe()
print(estadisticas_num_pen)

# Gráfico combinado - Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(df['Núm. pensiones'].values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
for median in bp['medians']:
    median.set(color='red', linewidth=2)
ax.text(1, estadisticas_num_pen['mean'] + 0.5, f'Media: {estadisticas_num_pen["mean"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_num_pen['min'] - 0.5, f'Mínimo: {estadisticas_num_pen["min"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_num_pen['max'] + 0.5, f'Máximo: {estadisticas_num_pen["max"]:.2f}', va='center', ha='left')
ax.set_xticks([])
plt.subplots_adjust(bottom=0.15, top=0.9, right=0.95)
plt.title('Boxplot de la variable Núm. pensiones')
plt.show()

# Gráfico de relación entre Año y Núm. pensiones con Seaborn
plt.figure(figsize=(8, 8))
sns.regplot(x="Año", y="Núm. pensiones", data=df, scatter_kws={"color": "red"}, line_kws={"color": "black"})
media_pib = df['Núm. pensiones'].mean()
for index, row in df.iterrows():
    if row['Núm. pensiones'] > media_pib:
        color = 'red'
    else:
        color = 'red'
    plt.scatter(row['Año'], row['Núm. pensiones'], color=color)
for index, row in df.iterrows():
    plt.text(row['Año'], row['Núm. pensiones']+0.5, f"{row['Núm. pensiones']:.0f}", ha='center', va='bottom')
plt.xticks(rotation=25)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
plt.title('Relación entre el Año y el Núm. pensiones')

plt.xlabel('Año')
plt.ylabel('Núm. pensiones')
plt.show()
