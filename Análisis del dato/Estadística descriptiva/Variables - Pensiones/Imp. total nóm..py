import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Lee los datos del archivo Excel
df = pd.read_excel("C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/data/altas&bajas_pensiones_sin2023.xlsx")

# Calcula las estadísticas descriptivas para cada columna
importe = df['Imp. total nóm.']
estadisticas_importe = importe.describe()
print(estadisticas_importe)

# Gráfico combinado - Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(df['Imp. total nóm.'].values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
for median in bp['medians']:
    median.set(color='red', linewidth=2)
ax.text(1, estadisticas_importe['mean'] + 0.5, f'Media: {estadisticas_importe["mean"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_importe['min'] - 0.5, f'Mínimo: {estadisticas_importe["min"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_importe['max'] + 0.5, f'Máximo: {estadisticas_importe["max"]:.2f}', va='center', ha='left')
ax.set_xticks([])
plt.subplots_adjust(bottom=0.15, top=0.9, right=0.95)
plt.title('Boxplot de la variable Imp. total nóm.')
plt.show()

# Gráfico de relación entre Año y Tasa de Desempleo con Seaborn
plt.figure(figsize=(8, 8))
sns.regplot(x="Año", y="Imp. total nóm.", data=df, scatter_kws={"color": "red"}, line_kws={"color": "black"})
media_pib = df['Imp. total nóm.'].mean()
for index, row in df.iterrows():
    color = 'red' if row['Imp. total nóm.'] > media_pib else 'red'
    plt.scatter(row['Año'], row['Imp. total nóm.'], color=color)
for index, row in df.iterrows():
    plt.text(row['Año'], row['Imp. total nóm.']+0.5, f"{row['Imp. total nóm.']:.0f}", ha='center', va='bottom')
plt.xticks(rotation=25)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
plt.title('Relación entre el Año y el Imp. total nóm.')
plt.xlabel('Año')
plt.ylabel('Imp. total nóm.')
plt.show()