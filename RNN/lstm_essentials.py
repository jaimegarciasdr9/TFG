import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow import keras
from tensorflow.keras import layers


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

def rnn_model(df, column_name, output_dir):
    returns = df[column_name].pct_change()
    print('PORCENTAJE DE CAMBIOS EN LA VARIABLE ALTAS PENSIONES')
    print(returns)
    plt.figure(figsize=(10, 6))
    plt.plot(returns)
    plt.title('Variación de la variable Altas pensiones')
    plt.xlabel('Año')
    plt.ylabel('Altas pensiones')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "variacion_altas_pensiones.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    returns.hist().plot()
    plt.title('Histograma variación Altas pensiones')
    plt.xlabel('Variación')
    plt.ylabel('Observaciones')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "histograma_variacion_altas_pensiones.png"))
    plt.close()
    print("-------------------------------------------------------")
    column_names = df.columns
    x = df.values
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    pensiones = pd.DataFrame(x_scaled)
    pensiones.columns = column_names
    print(pensiones)
    print("-------------------------------------------------------")
    print("FLATTEN MATRIX DOWN")
    npa = returns.values[1:].reshape(-1, 1)
    print(len(npa))
    print("Vamos a escalar los datos -- esto ayuda a evitar el problema de la explosión del gradiente")
    scale = MinMaxScaler(feature_range=(0, 1))
    npa = scale.fit_transform(npa)
    print(len(npa))

    print("-------------------------------------------------------")
    print("Calculate the time series data")
    samples = 10 # Usaré 10 del pasado
    # steps = 1 Para predecir un valor futuro
    X = []
    Y = []
    for i in range(npa.shape[0] - samples):
        X.append(npa[i:i+samples])  # Independent Samples
        Y.append(npa[i+samples][0]) # Dependent Samples
    print('Training Data: Length is', len(X[0:1][0]), ': ', X[0:1])
    print('Testing Data: Length is', len(Y[0:1]), ': ', Y[0:1])

    print("-------------------------------------------------------")
    print("Reshape the data so that the inputs will be acceptable to the model")
    X = np.array(X)
    Y = np.array(Y)
    print('Dimensions of X', X.shape, 'Dimensiones of Y', Y.shape)
    threshold = round(0.9 * X.shape[0])
    print('Threshold for training is', threshold)
    print('The rest is for testing')

    print("-------------------------------------------------------")
    print("Ahora creamos nuestra Arquitectura para nuestro modelo RNN")
    model = keras.Sequential()
    model.add(layers.SimpleRNN(3,
                               activation = 'tanh',
                               use_bias=True,
                               input_shape=(X.shape[1], X.shape[2])))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    print("-------------------------------------------------------")
    print("Fitting the data")
    history = model.fit(X[:threshold],
                        Y[:threshold],
                        shuffle=False,  # Since this is time series data
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1)  # Verbose outputs data
    print(history)

    print("-------------------------------------------------------")
    print("Trazado de la iteración de pérdidas")
    plt.figure(figsize=(10, 6))
    plt.title('Trazado de la iteración de pérdidas')
    plt.xlabel('Iteración')
    plt.ylabel('Pérdida')
    plt.plot(history.history['loss'], label='pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='pérdida validación')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "trazado_iteracion_perdidas.png"))
    plt.close()
    print("Nota")
    print("si pérdida de entrenamiento es mucho más grande que la pérdida de validación -> Infraajuste")
    print("si pérdida de entrenamiento es mucho más pequeña que la pérdida de validación -> Sobreajuste (es decir, el modelo es lo suficientemente inteligente como para haber mapeado todo el conjunto de datos..)")
    print("Varias maneras de abordar el sobreajuste:")
    print("Reducir la complejidad del modelo (capas ocultas, neuronas, parámetros de entrada, etc.).")
    print("Añadir tasa de abandono y ajuste")

    print("-------------------------------------------------------")
    print("Predicciones utilizando el enfoque de la ventana deslizante")
    true_Y = Y[threshold:]
    pred_Y = []
    print('Number of Forecasts to do: ', Y.shape[0] - round(Y.shape[0] * 0.9))
    latest_input = X[threshold - 1:threshold]
    for i in range(Y.shape[0] - round(Y.shape[0] * 0.9)):
        p = model.predict(latest_input.reshape(1, X.shape[1], 1))[0, 0]
        pred_Y.append(p)
        latest_input = np.append(X[threshold][1:], p)

    plt.figure(figsize=(10, 6))
    plt.title('Predicción por etapas múltiples')
    plt.xlabel('Etapas')
    plt.ylabel('Valor')
    plt.plot(true_Y, label='True Value')
    plt.plot(pred_Y, label='Forecasted Value')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "prediccion_multiples.png"))
    plt.close()

    print("Model just copied the same value over and over again. Hence, model is not very robust.")
    print("It's just predicting the mean.")

    print("Previsión multivariable. Utilizando nuestros conjuntos de entrenamiento y prueba, creemos nuestras entradas.")

    # Need the data to be in the form [sample, time steps, features (dimension of each element)]
    samples = 10  # Number of samples (in past)
    steps = 1  # Number of steps (in future)
    X = []  # X array
    Y = []  # Y array
    for i in range(df.shape[0] - samples):
        X.append(df.iloc[i:i + samples, 0:5].values)  # Independent Samples
        Y.append(df.iloc[i + samples, 5:].values)  # Dependent Samples
    print('Training Data: Length is ', len(X[0:1][0]), ': ', X[0:1])
    print('Testing Data: Length is ', len(Y[0:1]), ': ', Y[0:1])

    print("-------------------------------------------------------")
    print("Reshape the data so that the inputs will be acceptable to the model.")
    X = np.array(X)
    Y = np.array(Y)
    print('Dimensions of X', X.shape, 'Dimensions of Y', Y.shape)

    print("-------------------------------------------------------")
    print("Get the training and testing set")
    returns = df[column_name].pct_change().dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    returns_scaled = scaler.fit_transform(returns)

    # Prepare the data for RNN
    X, y = [], []
    for i in range(len(returns_scaled) - 5):
        X.append(returns_scaled[i:i + 5])
        y.append(returns_scaled[i + 5])
    X, y = np.array(X), np.array(y)

    # Split the data into train and test sets
    split = int(len(X) * 0.9)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # Define the RNN model
    model = keras.Sequential([
        layers.SimpleRNN(128, activation='relu', return_sequences=True,
                         input_shape=(X_train.shape[1], X_train.shape[2])),
        layers.SimpleRNN(64, activation='relu', return_sequences=True),
        layers.SimpleRNN(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=0)
    print('Test Loss:', loss)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Mean Squared Error (MSE):', mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    print('Mean Absolute Error (MAE):', mae)
    print('R-squared (R2):', r2)

    # Plot predictions vs true values
    plt.figure(figsize=(14, 4))
    plt.title('Predicciones frente a valores reales')
    plt.xlabel('Pasos de tiempo')
    plt.ylabel('Valores escalados de retorno')
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred, label='Predictions')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_true.png'))
    plt.close()

    # Obtener el índice de las últimas filas del DataFrame
    start_index = len(df) - len(y_pred)

    # Crear la columna 'pred_altas_pensiones' si no existe
    if 'pred_altas_pensiones' not in df.columns:
        df['pred_altas_pensiones'] = np.nan

    # Asignar las predicciones al DataFrame en las filas correspondientes
    df.iloc[start_index:, df.columns.get_loc('pred_altas_pensiones')] = y_pred

    # Imprimir las predicciones
    print("Predicciones de altas pensiones:")
    print(df['pred_altas_pensiones'])

    return pensiones, history







def main(df, column_name, output_dir):
    check_missing_values(df)
    create_output_dir(output_dir)
    rnn_model(df, column_name, output_dir)




if __name__ == "__main__":
    print("Este script no debe ejecutarse directamente.")
    print("Está pensado para ser ejecutado en otro script en paralelo")
