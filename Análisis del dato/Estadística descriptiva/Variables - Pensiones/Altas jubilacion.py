import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Lee los datos del archivo Excel
df = pd.read_excel("C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/data/altas&bajas_pensiones_sin2023.xlsx")

# Calcula las estadísticas descriptivas para cada columna
alt_jub = df['Altas jubilación']
estadisticas_alt_jub = alt_jub.describe()
print(estadisticas_alt_jub)

# Gráfico combinado - Boxplot
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(df['Altas jubilación'].values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
for median in bp['medians']:
    median.set(color='red', linewidth=2)
ax.text(1, estadisticas_alt_jub['mean'] + 0.5, f'Media: {estadisticas_alt_jub["mean"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_alt_jub['min'] - 0.5, f'Mínimo: {estadisticas_alt_jub["min"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_alt_jub['max'] + 0.5, f'Máximo: {estadisticas_alt_jub["max"]:.2f}', va='center', ha='left')
ax.set_xticks([])
plt.subplots_adjust(bottom=0.15, top=0.9, right=0.95)
plt.title('Boxplot de la variable Altas jubilación')
plt.show()

# Gráfico de relación entre Año y Tasa de Desempleo con Seaborn
plt.figure(figsize=(8, 8))
sns.regplot(x="Año", y="Altas jubilación", data=df, scatter_kws={"color": "red"}, line_kws={"color": "black"})
media_pib = df['Altas jubilación'].mean()
for index, row in df.iterrows():
    color = 'red' if row['Altas jubilación'] > media_pib else 'red'
    plt.scatter(row['Año'], row['Altas jubilación'], color=color)
for index, row in df.iterrows():
    plt.text(row['Año'], row['Altas jubilación']+0.5, f"{row['Altas jubilación']:.0f}", ha='center', va='bottom')
plt.xticks(rotation=25)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
plt.title('Relación entre el Año y las Altas jubilación')
plt.xlabel('Año')
plt.ylabel('Altas jubilación')
plt.show()