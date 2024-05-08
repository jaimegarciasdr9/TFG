import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

warnings.filterwarnings("ignore")


def load_data(file_path):
    df = pd.read_excel(file_path)
    df['año_mes'] = pd.to_datetime(df['año_mes'])
    df.set_index('año_mes', inplace=True)
    df.index.freq = 'MS'
    return df


def check_missing_values(df):
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print("Se encontraron valores faltantes en el DataFrame:")
        print(df.isnull().sum())
        return True
    else:
        print("No se encontraron valores faltantes en el DataFrame.")
        return False


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio {output_dir} creado con éxito.")
    else:
        print(f"El directorio {output_dir} ya existe.")


def test_stationarity(df, column_name, output_dir):
    """
    Realiza original de estacionariedad en una serie temporal y visualiza los resultados.
    Parámetros:
    - df: DataFrame que contiene los datos.
    - column_name: Nombre de la columna a evaluar.
    - output_dir: Directorio donde se guardarán las imágenes.
    """
    datos_diff_1 = df[column_name].diff().dropna()
    print('--------------------------------------------------')
    print('Test estacionariedad serie original')
    adfuller_result = adfuller(df[column_name])
    kpss_result = kpss(df[column_name])
    print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
    print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')
    print('--------------------------------------------------')
    print('\nTest estacionariedad para serie diferenciada (order=1)')
    adfuller_result = adfuller(datos_diff_1)
    kpss_result = kpss(df[column_name].diff().dropna())
    print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
    print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')
    # Gráfico series
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), sharex=True)
    df[column_name].plot(ax=axs[0], title='Serie original')
    datos_diff_1.plot(ax=axs[1], title='Diferenciación orden 1')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{column_name}_primera_diferenciacion.png"))
    plt.close()  # Cierra la figura después de guardar
    return datos_diff_1


def plot_acf_comparison(df, column_name, output_dir):
    """
    Grafica la autocorrelación para la serie original y la serie diferenciada.
    Parámetros:
    - df: DataFrame que contiene los datos.
    - column_name: Nombre de la columna a evaluar.
    - output_dir: Directorio donde se guardarán las imágenes.
    """
    datos_diff_1 = df[column_name].diff().dropna()  # Se eliminan los valores NaN después de la diferenciación
    num_lags_original = len(df[column_name]) - 1
    num_lags_diferenciada = len(datos_diff_1) - 1
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    plot_acf(df[column_name], ax=axs[0], lags=num_lags_original, alpha=0.05,
             title=f'Autocorrelación serie original ({num_lags_original} lags)')
    plot_acf(datos_diff_1, ax=axs[1], lags=num_lags_diferenciada, alpha=0.05,
             title=f'Autocorrelación serie diferenciada (order=1) ({num_lags_diferenciada} lags)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "autocorrelacion_comparison.png"))
    plt.close()


def plot_pacf_comparison(df, column_name, output_dir):
    """
    Grafica la autocorrelación parcial para la serie original y la serie diferenciada.
    Parámetros:
    - df: DataFrame que contiene los datos.
    - column_name: Nombre de la columna a evaluar.
    - output_dir: Directorio donde se guardarán las imágenes.
    """
    datos_diff_1 = df[column_name].diff().dropna()  # Se eliminan los valores NaN después de la diferenciación
    max_lags_original = int(len(df[column_name]) * 0.5)
    max_lags_diferenciada = int(len(datos_diff_1) * 0.5)
    num_lags_original = min(50, max_lags_original)
    num_lags_diferenciada = min(50, max_lags_diferenciada)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    plot_pacf(df[column_name], ax=axs[0], lags=num_lags_original,
              alpha=0.05, title=f'Autocorrelación parcial serie original ({num_lags_original} lags)')
    plot_pacf(datos_diff_1, ax=axs[1], lags=num_lags_diferenciada,
              alpha=0.05, title=f'Autocorrelación parcial serie diferenciada (order=1) ({num_lags_diferenciada} lags)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "autocorrelacion_parcial_comparison.png"))
    plt.close()


def plot_decomposition_comparison(df, column_name, output_dir):
    """
    Realiza la descomposición de la serie original y la serie diferenciada.
    Parámetros:
    - df: DataFrame que contiene los datos.
    - column_name: Nombre de la columna a evaluar.
    - output_dir: Directorio donde se guardarán las imágenes.
    """
    datos_diff_1 = df[column_name].diff().dropna()
    res_decompose = seasonal_decompose(df[column_name], model='additive', period=6, extrapolate_trend='freq')
    res_decompose_diff_1 = seasonal_decompose(datos_diff_1, model='additive', period=6, extrapolate_trend='freq')
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 6), sharex=True)
    res_decompose.observed.plot(ax=axs[0, 0])
    axs[0, 0].set_title('Serie original')
    res_decompose.trend.plot(ax=axs[1, 0])
    axs[1, 0].set_title('Tendencia')
    res_decompose.seasonal.plot(ax=axs[2, 0])
    axs[2, 0].set_title('Estacionalidad')
    res_decompose.resid.plot(ax=axs[3, 0])
    axs[3, 0].set_title('Residuos')
    res_decompose_diff_1.observed.plot(ax=axs[0, 1])
    axs[0, 1].set_title('Serie diferenciada (order=1)')
    res_decompose_diff_1.trend.plot(ax=axs[1, 1])
    axs[1, 1].set_title('Tendencia')
    res_decompose_diff_1.seasonal.plot(ax=axs[2, 1])
    axs[2, 1].set_title('Estacionalidad')
    res_decompose_diff_1.resid.plot(ax=axs[3, 1])
    axs[3, 1].set_title('Residuos')
    fig.suptitle('Descomposición de la serie original vs serie diferenciada', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "descomposicion_comparison.png"))
    plt.close()


def differencing_order_1_seasonal_differencing(df, column_name):
    """
    Realiza diferenciación de orden 1 combinada con diferenciación estacional.
    Parámetros:
    - datos_train: Serie temporal de entrenamiento.
    Retorna:
    - datos_diff_1_12: Serie resultante después de aplicar diferenciación de orden 1 combinada con diferenciación estacional.
    """
    datos_diff_1_12 = df[column_name].diff().diff(12).dropna()
    adfuller_result = adfuller(datos_diff_1_12)
    print('--------------------------------------------------')
    print('DIFERENCIACIÓN DE ORDEN 1')
    print(f'ADF Statistic: {adfuller_result[0]}, p-value: {adfuller_result[1]}')
    kpss_result = kpss(datos_diff_1_12)
    print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')
    return datos_diff_1_12


def visualize_data(df, column_name, output_dir):
    fin_train = df.index[int(len(df) * 0.95)]  # Por ejemplo, puedo usar el 80% de los datos para entrenamiento
    print('--------------------------------------------------')
    print('FECHAS TRAIN/TEST')
    print(f"Fechas train : {df.index.min()} --- {df.loc[:fin_train].index.max()}  (n={len(df.loc[:fin_train])})")
    print(f"Fechas test  : {df.loc[fin_train:].index.min()} --- {df.index.max()}  (n={len(df.loc[fin_train:])})")
    datos_train = df.loc[:fin_train]
    datos_test = df.loc[fin_train:]
    plt.figure(figsize=(10, 6))
    plt.plot(datos_train.index, datos_train[column_name], color='green', linestyle='--', label='Train')
    plt.plot(datos_test.index, datos_test[column_name], color='red', linestyle='--', label='Test')
    plt.title('Altas Pensiones - Serie Temporal')
    plt.xlabel('Año')
    plt.ylabel('Altas Pensiones')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{column_name}_grafico_train_test.png"))
    plt.close()


def auto_arima_model(df, column_name):
    """
    Ajusta automáticamente un modelo ARIMA utilizando el método auto_arima de pmdarima.
    Parámetros:
    - df: DataFrame que contiene los datos de la serie temporal.
    - column_name: Nombre de la columna que contiene la serie temporal.
    Retorna:
    - stepwise_fit: Resumen del modelo ARIMA ajustado automáticamente.
    """
    stepwise_fit = auto_arima(df[column_name], trace=True, suppress_warnings=True)
    print('--------------------------------------------------')
    print('RESUMEN DEL MODELO ARIMA AJUSTADO AUTOMÁTICAMENTE')
    print(stepwise_fit.summary())
    return stepwise_fit


def arima_model(df, column_name, output_dir):
    """
    Ajusta el modelo ARIMA y realiza predicciones.
    Parámetros:
    - df: DataFrame que contiene los datos de la serie temporal.
    - column_name: Nombre de la columna que contiene la serie temporal.
    - output_dir: Directorio de salida para guardar las gráficas.
    Retorna:
    - model_fit: Resumen del modelo ARIMA ajustado.
    """
    # Realiza la diferenciación de orden 1 combinada con diferenciación estacional
    datos_diff_1 = df[column_name].diff().dropna()
    # Ajusta el modelo ARIMA
    model = ARIMA(datos_diff_1, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
    # Realizar predicciones de la serie temporal
    residuals = model_fit.resid[1:]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Ajusta el tamaño de la figura aquí
    residuals.plot(title='Residuos', ax=ax[0])
    residuals.plot(title='Densidad', kind='kde', ax=ax[1])
    ax[0].set_xlabel('Índice residual')
    ax[0].set_ylabel('Valor residual')
    ax[1].set_xlabel('Valor residual')
    ax[1].set_ylabel('Densidad')
    plt.savefig(os.path.join(output_dir, "modelo_arima.png"))
    plt.close()
    return model_fit


def train_arima_model(df, column_name):
    # División de los datos en conjuntos de entrenamiento y prueba
    datos_reales = df[column_name]
    fin_train_index = int(len(df) * 0.96)
    datos_train = datos_reales.iloc[:fin_train_index]
    datos_test = datos_reales.iloc[fin_train_index:]

    # Ajuste del modelo ARIMA
    model = auto_arima(datos_train, seasonal=False, suppress_warnings=True, stepwise=True)
    model_fit = model.fit(datos_train)

    # Obtención de las predicciones
    model_predictions = model_fit.predict(n_periods=len(datos_test))  # Obtener todas las predicciones de una sola vez

    # Cálculo de métricas
    mse = mean_squared_error(datos_test, model_predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(datos_test, model_predictions)
    r2 = r2_score(datos_test, model_predictions)

    print('--------------------------------------------------')
    print('RESUMEN MÉTRICAS DEL MODELO')
    print(model_fit.summary())
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"R cuadrado: {r2:.4f}")
    return model_fit, datos_train, datos_test, model_predictions


def plot_predictions(model_predictions, datos_test, output_dir):
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.plot(datos_test, label="Datos reales")
    plt.plot(model_predictions, label="Predicciones")
    plt.title('Comparación de Predicciones y Datos Reales Altas pensiones')
    plt.xlabel('Año')
    plt.ylabel('Altas pensiones')  # Usamos el nombre de la columna
    plt.legend()
    plt.savefig(os.path.join(output_dir, "predicciones_modelo_arima_comparacion.png"))
    plt.close()

def plot_diagnostics(model_fit, output_dir):
    plt.style.use('ggplot')
    diag_plots = model_fit.plot_diagnostics(figsize=(10, 10))
    for ax in diag_plots.axes:
        for line in ax.lines:
            line.set_color('blue')
        for patch in ax.patches:
            patch.set_color('blue')
    plt.savefig(os.path.join(output_dir, "diagnostics.png"))
    plt.close()


def calculate_metrics(df, model_predictions, column_name):
    datos_reales = df[column_name]
    model_predictions = model_predictions[:len(datos_reales)]
    # Calcular el error absoluto entre los datos reales y las predicciones
    error_absoluto = np.abs(datos_reales - model_predictions)
    # Calcular el error cuadrático entre los datos reales y las predicciones
    error_cuadratico = (datos_reales - model_predictions) ** 2
    # Calcular la desviación estándar de los errores absolutos
    std_error_absoluto = np.std(error_absoluto)
    # Calcular el máximo error absoluto
    max_error_absoluto = np.max(error_absoluto)
    # Calcular el máximo error cuadrático
    max_error_cuadratico = np.max(error_cuadratico)
    # Calcular el error absoluto promedio
    mean_error_absoluto = np.mean(error_absoluto)
    # Calcular el error cuadrático promedio
    mean_error_cuadratico = np.mean(error_cuadratico)

    print('--------------------------------------------------')
    print('RESUMEN MÉTRICAS DEL MODELO')
    print(f"Desviación estándar del error absoluto: {std_error_absoluto:.2f}")
    print(f"Máximo error absoluto: {max_error_absoluto:.2f}")
    print(f"Máximo error cuadrático: {max_error_cuadratico:.2f}")
    print(f"Error absoluto promedio: {mean_error_absoluto:.2f}")
    print(f"Error cuadrático promedio: {mean_error_cuadratico:.2f}")

def main(df, column_name, output_dir):
    visualize_data(df, column_name, output_dir)
    test_stationarity(df, column_name, output_dir)
    plot_acf_comparison(df, column_name, output_dir)
    plot_pacf_comparison(df, column_name, output_dir)
    plot_decomposition_comparison(df, column_name, output_dir)
    auto_arima_model(df, column_name)
    model_fit, datos_train, datos_test, model_predictions = train_arima_model(df, column_name)
    plot_diagnostics(model_fit, output_dir)
    plot_predictions(model_predictions, datos_test, output_dir)
    calculate_metrics(df, model_predictions, column_name)


if __name__ == "__main__":
    print("Este script no debe ejecutarse directamente.")

