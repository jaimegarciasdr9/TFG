import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd


dataset = pd.read_excel("C:/Users/jaime/Desktop/TFG/Presentación Análisis del dato/data/altas&bajas_pensiones_sin2023.xlsx")



# Paso 1: Preparación de los datos
# Dividir los datos en secuencias de entrada y salida
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)


fin_train = '2018'
seq_length = 18

# Imprimir información sobre las fechas de entrenamiento y prueba
print('--------------------------------------------------')
print('FECHAS TRAIN/TEST')
print(f"Fechas train : {dataset.index.min()} --- {dataset.loc[:fin_train].index.max()}  (n={len(dataset.loc[:fin_train])})")
print(f"Fechas test  : {dataset.loc[fin_train:].index.min()} --- {dataset.index.max()}  (n={len(dataset.loc[fin_train:])})")

# Obtener datos de entrenamiento y prueba
datos_train = dataset.loc[:fin_train]
datos_test = dataset.loc[fin_train:]

# Crear secuencias de entrenamiento y prueba
X_train, y_train = create_sequences(datos_train, seq_length)
X_test, y_test = create_sequences(datos_test, seq_length)

# Paso 2: Normalización de los datos
scaler = MinMaxScaler()
scaler.fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)

# Paso 3: Construcción del modelo RNN
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

# Agregar una dimensión adicional a los datos de entrenamiento y prueba
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Paso 4: Compilación y entrenamiento del modelo
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Paso 5: Evaluación y predicción
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print('Train Loss:', train_loss)
print('Test Loss:', test_loss)

# Predicción
predictions = model.predict(X_test)
print(predictions)
