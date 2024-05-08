import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Año'] = df['Año'].astype(int)
    df.set_index('Año', inplace=True)
    return df

def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio {output_dir} creado con éxito.")
    else:
        print(f"El directorio {output_dir} ya existe.")

def print_descriptive_statistics(df, column_name):
    print(f"Estadísticas descriptivas para la columna '{column_name}':")
    statistics = df[column_name].describe()
    print(statistics)


def plot_time_series(df, column_name, output_dir, label_height):
    """Traza la serie temporal"""
    plt.figure(figsize=(10, 6))
    ax = df[column_name].plot(color='blue', linewidth=2)
    plt.title(f'Serie temporal - {column_name}')
    plt.xlabel('Año')
    plt.ylabel(column_name)
    formatter = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x))
    plt.gca().xaxis.set_major_formatter(formatter)
    # Agregar línea vertical intermitente en el año 2020 y etiqueta
    ax.axvline(x=2020, color='orange', linestyle='--')
    plt.text(2020.1, df[column_name].max() * label_height, 'Covid-19', color='orange', fontsize=8, rotation=90, va='bottom')
    plt.savefig(os.path.join(output_dir, f"{column_name}_serie_temporal.png"))

def plot_scatter_plot(df, column_name, output_dir, label_height):
    """Dibujar un diagrama de puntos (scatter plot)"""
    plt.figure(figsize=(10, 6))
    sns.regplot(x=df.index, y=df[column_name], scatter_kws={"color": "blue"}, line_kws={"color": "black"})
    media_pib = df[column_name].mean()
    for index, row in df.iterrows():
        color = 'blue' if row[column_name] > media_pib else 'blue'
        plt.scatter(index, row[column_name], color=color)  # Corrección aquí
    for index, row in df.iterrows():
        plt.text(index, row[column_name]+0.5, f"{row[column_name]:.0f}", ha='center', va='bottom')  # Corrección aquí
    plt.xticks(rotation=25)
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.0f}'))
    plt.title(f'Relación entre el Año y {column_name}')
    plt.xlabel('Año')
    plt.ylabel(column_name)
    # Agregar línea vertical intermitente en el año 2020 y etiqueta
    plt.axvline(x=2020, color='orange', linestyle='--')
    plt.text(2020.1, df[column_name].max() * label_height, 'Covid-19', color='orange', fontsize=8, rotation=90, va='bottom')
    plt.savefig(os.path.join(output_dir, f"{column_name}_diagrama_dispersion.png"))

def plot_area_plot(df, output_dir, column_name, column_name_2, label_height):
    """Dibujar el área comprendida entre dos variables"""
    plt.figure(figsize=(10, 6))
    df[column_name].plot(color='green', linewidth=2)
    plt.fill_between(df.index, df[column_name], color='lightgreen', alpha=0.3, label=column_name)
    df[column_name_2].plot(color='red', linewidth=2)  # Cambio el color a rojo
    plt.fill_between(df.index, df[column_name_2], color='lightcoral', alpha=0.3, label=column_name_2)  # Cambio a lightcoral para el área
    plt.title(f'Serie temporal - {column_name} y {column_name_2}')
    plt.xlabel('Año')
    plt.ylabel(f'{column_name} - {column_name_2}')
    formatter = ticker.FuncFormatter(lambda x, pos: '{:.0f}'.format(x))
    plt.gca().xaxis.set_major_formatter(formatter)
    # Agregar línea vertical intermitente en el año 2020 y etiqueta
    plt.axvline(x=2020, color='orange', linestyle='--')
    plt.text(2020.1, max(df[column_name].max(), df[column_name_2].max()) * label_height, 'Covid-19', color='orange', fontsize=8, rotation=90, va='bottom')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{column_name}_{column_name_2}_area_debajo_curva.png"))

def plot_boxplot(df, output_dir, column_name):
    """Gráfico combinado - Boxplot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(df[column_name].values, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    ax.text(1.10, df[column_name].mean() + 0.5, f'Media: {df[column_name].mean():.2f}', va='center', ha='left')
    ax.text(1.05, df[column_name].min() - 0.5, f'Mínimo: {df[column_name].min():.2f}', va='center', ha='left')
    ax.text(1.05, df[column_name].max() + 0.5, f'Máximo: {df[column_name].max():.2f}', va='center', ha='left')
    ax.set_xticks([])
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.95)
    plt.title(f'Boxplot de la variable {column_name}')
    plt.savefig(os.path.join(output_dir, f"{column_name}_boxplot.png"))

def plot_rolling_mean(df, column_name, output_dir):
    """Gráfico de la media móvil por periodos anuales"""
    plt.figure(figsize=(10, 6))
    window_size = len(df.index) * 3 // 20  # Utilizar dos tercios del tamaño del índice
    df[column_name].rolling(window=window_size).mean().plot(color="tab:blue")
    plt.title(f"Media móvil por periodos anuales en {column_name}")
    plt.xlabel("Año")
    plt.ylabel(f"Media móvil de {column_name}")
    plt.savefig(os.path.join(output_dir, f"media_movil_{column_name}.png"))

def plot_rolling_variance(df, column_name, output_dir):
    """Gráfico de la varianza móvil por periodos anuales"""
    plt.figure(figsize=(10, 6))
    window_size = len(df.index) * 3 // 20  # Utilizar dos tercios del tamaño del índice
    df[column_name].rolling(window=window_size).var().plot(color="tab:blue")
    plt.title(f"Varianza móvil por periodos anuales en {column_name}")
    plt.xlabel("Año")
    plt.ylabel("Varianza móvil")
    plt.savefig(os.path.join(output_dir, f"varianza_movil_{column_name}.png"))

def plot_decomposition(df, column_name, output_dir):
    """Gráfico de descomposición por periodos"""
    plt.figure(figsize=(10, 6))
    window_size = len(df.index) * 3 // 25
    decompose_result = seasonal_decompose(df[column_name], model='additive', period=window_size, extrapolate_trend='freq')
    decompose_result.plot()
    plt.savefig(os.path.join(output_dir, f"funcion_descomposicion_{column_name}.png"))

def test_stationarity(df, column_name):
    """Pruebas estacionarias con Dickey-Fuller"""
    window_size = len(df.index) * 3 // 25
    decompose_result = seasonal_decompose(df[column_name], model='additive', period=window_size, extrapolate_trend='freq')
    residual = decompose_result.resid
    dftest = adfuller(residual, autolag='AIC')
    print("1. ADF:", dftest[0])
    print("2. P-Value:", dftest[1])
    print("3. Num Of Lags:", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation:", dftest[3])
    print("5. Critical Values:")
    for key, val in dftest[4].items():
        print("\t", key, ":", val)




def main(df, column_name, output_dir, label_height, column_name_2):
    print_descriptive_statistics(df, column_name)
    plot_time_series(df, column_name, output_dir, label_height)
    plot_scatter_plot(df, column_name, output_dir, label_height)
    plot_area_plot(df, output_dir, column_name, column_name_2, label_height)
    plot_boxplot(df, output_dir, column_name)
    plot_rolling_mean(df, column_name, output_dir)
    plot_rolling_variance(df, column_name, output_dir)
    plot_decomposition(df, column_name, output_dir)
    test_stationarity(df, column_name)


if __name__ == "__main__":
    print("Este script no debe ejecutarse directamente.")