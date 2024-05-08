import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Lee los datos del archivo Excel
df = pd.read_excel("C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/data/altas&bajas_pensiones_sin2023.xlsx")
df['Año'] = df['Año'].astype(int)
df.set_index('Año', inplace=True)

# Directorio para guardar la imagen
output_dir = "C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/graficos/altas_pensiones"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""Texto con la información"""
print("Valores nulos por columna:")
print(df.isnull().sum())
print("Estadísticas descriptivas")
estadisticas_alt_pen = df['Altas pensiones'].describe()
print("Información del DataFrame")
info_alt_pen = df['Altas pensiones'].info()

"""Traza la serie temporal"""
plt.figure(figsize=(20, 4))
ax = df['Altas pensiones'].plot(color='blue', linewidth=2)
plt.title('Serie temporal - Altas en las pensiones')
plt.xlabel('Año')
plt.ylabel('Pensiones nuevas')
formatter = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x))
plt.gca().xaxis.set_major_formatter(formatter)
# Agregar línea vertical intermitente en el año 2020 y etiqueta
ax.axvline(x=2020, color='orange', linestyle='--')
plt.text(2020.1, df['Altas pensiones'].max() * 0.95, 'Covid-19', color='orange', fontsize=8, rotation=90, va='bottom')
plt.savefig(os.path.join(output_dir, "serie_temporal.png"))  # Guarda el gráfico como 'serie_temporal.png'


"""Dibujar un diagrama de puntos (scatter plot)"""
plt.figure(figsize=(20, 10))
sns.regplot(x=df.index, y=df['Altas pensiones'], scatter_kws={"color": "blue"}, line_kws={"color": "black"})
media_pib = df['Altas pensiones'].mean()
for index, row in df.iterrows():
    color = 'blue' if row['Altas pensiones'] > media_pib else 'blue'
    plt.scatter(index, row['Altas pensiones'], color=color)  # Corrección aquí
for index, row in df.iterrows():
    plt.text(index, row['Altas pensiones']+0.5, f"{row['Altas pensiones']:.0f}", ha='center', va='bottom')  # Corrección aquí
plt.xticks(rotation=25)
plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
plt.title('Relación entre el Año y las Altas en pensiones')
plt.xlabel('Año')
plt.ylabel('Altas pensiones')
# Agregar línea vertical intermitente en el año 2020 y etiqueta
plt.axvline(x=2020, color='orange', linestyle='--')
plt.text(2020.1, df['Altas pensiones'].max() * 0.85, 'Covid-19', color='orange', fontsize=8, rotation=90, va='bottom')
plt.savefig(os.path.join(output_dir, "diagrama_dispersion.png"))  # Guarda el gráfico como 'diagrama_dispersion.png'


"""Dibujar el área"""
plt.figure(figsize=(20, 10))
ax = df['Altas pensiones'].plot(color='blue', linewidth=2)
plt.fill_between(df.index, df['Altas pensiones'], color='lightblue', alpha=0.3, label='Altas pensiones')  # Rellenar el área debajo de la curva de altas pensiones
ax = df['Bajas pensiones'].plot(color='green', linewidth=2)
plt.fill_between(df.index, df['Bajas pensiones'], color='lightgreen', alpha=0.3, label='Bajas pensiones')  # Rellenar el área debajo de la curva de bajas pensiones
plt.title('Serie temporal - Altas y Bajas en las pensiones')
plt.xlabel('Año')
plt.ylabel('Altas - Bajas Pensiones')
formatter = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x))
plt.gca().xaxis.set_major_formatter(formatter)
# Agregar línea vertical intermitente en el año 2020 y etiqueta
ax.axvline(x=2020, color='orange', linestyle='--')
plt.text(2020.1, df[['Altas pensiones', 'Bajas pensiones']].max().max() * 0.95, 'Covid-19', color='orange', fontsize=8, rotation=90, va='bottom')
plt.legend()  # Mostrar leyenda
plt.savefig(os.path.join(output_dir, "area_debajo_curva.png"))  # Guarda el gráfico como 'area_debajo_curva.png'


"""Gráfico combinado - Boxplot"""
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot(df['Altas pensiones'].values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
for median in bp['medians']:
    median.set(color='red', linewidth=2)
ax.text(1, estadisticas_alt_pen['mean'] + 0.5, f'Media: {estadisticas_alt_pen["mean"]:.2f}', va='center', ha='left')
ax.text(1, estadisticas_alt_pen['min'] - 0.5, f'Mínimo: {estadisticas_alt_pen["min"]:.0f}', va='center', ha='left')
ax.text(1, estadisticas_alt_pen['max'] + 0.5, f'Máximo: {estadisticas_alt_pen["max"]:.0f}', va='center', ha='left')
ax.set_xticks([])
plt.subplots_adjust(bottom=0.15, top=0.9, right=0.95)
plt.title('Boxplot de la variable Altas pensiones')
plt.savefig(os.path.join(output_dir, "diagrama_de_cajas.png"))


"""Matriz de correlación entre todas las columnas"""
correlation_matrix = df.corr()
plt.figure(figsize=(24, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación entre todas las variables")
plt.xlabel("Variables")
plt.ylabel("Variables")
plt.savefig(os.path.join(output_dir, "matriz_correlacion.png"))
