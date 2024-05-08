import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Leer datos desde un archivo Excel
archivo_excel = 'C:/Users/jaime/Desktop/TFG/data/Presentación Análisis del dato/altas&bajas_pensiones.xlsx'
datos = pd.read_excel(archivo_excel, sheet_name='Hoja1')

# Verificar los tipos de datos en las columnas
print(datos.dtypes)

# Convertir columnas a tipos numéricos
columnas_numericas = ['Orden', 'Año', 'Clase pensión', 'Altas pensiones', 'Bajas pensiones']

for columna in columnas_numericas:
    datos[columna] = pd.to_numeric(datos[columna], errors='coerce')

# Estadísticas descriptivas
print(datos.describe())

media = np.mean(datos['Altas pensiones'].dropna())  # Usamos dropna() para eliminar NaN antes del cálculo
print("Media:", media)
mediana = np.median(datos['Altas pensiones'].dropna())
print("Mediana:", mediana)
varianza = np.var(datos['Altas pensiones'].dropna())
print("Varianza:", varianza)
desviacion_estandar = np.std(datos['Altas pensiones'].dropna())
print("Desviación estándar:", desviacion_estandar)
minimo = np.min(datos['Altas pensiones'].dropna())
print("Mínimo:", minimo)

maximo = np.max(datos['Altas pensiones'].dropna())
print("Máximo:", maximo)
percentil_25 = np.percentile(datos['Altas pensiones'].dropna(), 25)
percentil_50 = np.percentile(datos['Altas pensiones'].dropna(), 50)  # = la mediana
percentil_75 = np.percentile(datos['Altas pensiones'].dropna(), 75)
print("Percentiles 25, 50 y 75:", percentil_25, percentil_50, percentil_75)
IQR = percentil_75 - percentil_25
print("Rango intercuartil (IQR):", IQR)

# Agrupar los datos por edad
datos_agrupados_por_edad = datos.groupby('Año')

# Crear el gráfico de histogramas
plt.figure(figsize=(10, 6))

# Trazar un histograma para cada grupo de edad
for edad, grupo in datos_agrupados_por_edad:
    plt.hist(grupo['Altas pensiones'].dropna(), bins=10, alpha=0.5, label=edad)

# Configuraciones del gráfico
plt.xlabel('Altas pensiones')
plt.ylabel('Frecuencia')
plt.title('Histograma de Altas pensiones por Año')
plt.legend(title='Año')
plt.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Crear el boxplot con Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x=datos['Altas pensiones'])
7
# Estadísticas descriptivas
descripcion = datos['Altas pensiones'].describe()

# Mostrar las estadísticas descriptivas
print("Estadísticas descriptivas para 'Altas pensiones':")
print(descripcion)

# Configuraciones del gráfico
plt.xlabel('Altas pensiones')
plt.title('Altas pensiones')

# Mostrar el gráfico
plt.grid(True)
plt.show()

# ¿Existen valores atípicos?
# Identificar y filtrar valores atípicos basados en un criterio específico
valor_atipico = 40000  # Definir el valor límite para considerar como anómalo

# Filtrar los valores atípicos que están por debajo del umbral
datos_limpios = datos[datos['Altas pensiones'] < valor_atipico]

# Calcular el resumen estadístico de los datos limpios
resumen_estadistico = datos_limpios['Altas pensiones'].describe()

# Mostrar el resumen estadístico
print(resumen_estadistico)

# Crear el boxplot con Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x=datos_limpios['Altas pensiones'])

# Estadísticas descriptivas
descripcion = datos_limpios['Altas pensiones'].describe()

# Mostrar las estadísticas descriptivas
print("Estadísticas descriptivas para 'Altas pensiones':")
print(descripcion)

# Configuraciones del gráfico
plt.xlabel('Altas pensiones')
plt.title('Altas pensiones')

# Mostrar el gráfico
plt.grid(True)
plt.show()

percentil_25 = np.percentile(datos_limpios['Altas pensiones'].dropna(), 25)
percentil_50 = np.percentile(datos_limpios['Altas pensiones'].dropna(), 50)  # = la mediana
percentil_75 = np.percentile(datos_limpios['Altas pensiones'].dropna(), 75)
print("Percentiles 25, 50 y 75:", percentil_25, percentil_50, percentil_75)
IQR = percentil_75 - percentil_25
print("Rango intercuartil (IQR):", IQR)
