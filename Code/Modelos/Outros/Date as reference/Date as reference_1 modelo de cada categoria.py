# Date as reference

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_excel('C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Skin_clean and treated/Skin_clean.xlsx')

# Converter a data para datetime e ordenar de forma crescente
df['Screening_date'] = pd.to_datetime(df['Screening_date'])
df = df.sort_values(by='Screening_date')

# Imputação de valores ausentes
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna(df[column].mode().iloc[0])

print(df)

# Definir uma data de referência
reference_date = pd.Timestamp('2022-01-01')

# Calcular a diferença em dias a partir da data de referência
df['Days_from_reference'] = (df['Screening_date'] - reference_date).dt.days

# Definir as features (X) e a variável target (y)
X = df.drop(columns=['Screening_date', 'Days_from_reference'])
y = df['Days_from_reference']

print(y.head(100))

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

######################## LINEAR REGRESSION ######################

# Criar o modelo de Regressão Linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
me = np.mean(y_test - y_pred)
mav = np.mean(np.abs(y_test))
mpv = np.mean(np.abs((y_test - y_pred) / y_test))
rme = np.mean((y_test - y_pred) / y_test)
rmae = np.mean(np.abs(y_test - y_pred) / np.abs(y_test))

print("Modelo: LR")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# # Visualizar as previsões vs os valores reais
# plt.figure(figsize=(10, 5))
# plt.plot(y_test.values, label='Real')
# plt.plot(y_pred, label='Previsto')
# plt.legend()
# plt.show()

#### results

# MAE: 201.8140126189182
# MSE: 57377.13771910972
# RMSE: 239.53525360395224
# R2: 0.07137554928711531
# ME: -9.142030109725559
# MAV: 515.473496128648
# MPV: 0.8198384506200553
# RME: -0.5675761700262302
# RMAE: 0.8198384506200553

############################# KNN ###########################

from sklearn.neighbors import KNeighborsRegressor

# Criar o modelo KNN
model = KNeighborsRegressor(n_neighbors=5)  # você pode ajustar o número de vizinhos (k) conforme necessário

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
me = np.mean(y_test - y_pred)
mav = np.mean(np.abs(y_test))
mpv = np.mean(np.abs((y_test - y_pred) / y_test))
rme = np.mean((y_test - y_pred) / y_test)
rmae = np.mean(np.abs(y_test - y_pred) / np.abs(y_test))

print("Modelo: KNN")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# # Visualizar as previsões vs os valores reais
# plt.figure(figsize=(10, 5))
# plt.plot(y_test.values, label='Real')
# plt.plot(y_pred, label='Previsto')
# plt.legend()
# plt.show()

#### results

# MAE: 197.08648004764737
# MSE: 58697.97786777844
# RMSE: 242.2766556393299
# R2: 0.0499983160144889
# ME: -5.391185229303156
# MAV: 515.473496128648
# MPV: 0.7714368574452231
# RME: -0.5017122960099533
# RMAE: 0.7714368574452231


############################# RF ###########################

from sklearn.ensemble import RandomForestRegressor

# Criar o modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)  # você pode ajustar o número de estimadores (árvores) conforme necessário

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
me = np.mean(y_test - y_pred)
mav = np.mean(np.abs(y_test))
mpv = np.mean(np.abs((y_test - y_pred) / (y_test + np.finfo(float).eps)))  # Adiciona um pequeno valor para evitar divisão por zero
rme = np.mean((y_test - y_pred) / (y_test + np.finfo(float).eps))  # Adiciona um pequeno valor para evitar divisão por zero
rmae = np.mean(np.abs(y_test - y_pred) / (np.abs(y_test) + np.finfo(float).eps))  # Adiciona um pequeno valor para evitar divisão por zero

print("Modelo: RF")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# Visualizar as previsões vs os valores reais
# plt.figure(figsize=(10, 5))
# plt.plot(y_test, label='Real')
# plt.plot(y_pred, label='Previsto')
# plt.legend()
# plt.show()

#### results

# MAE: 183.0943763714929
# MSE: 54472.75312661996
# RMSE: 233.3939869118739
# R2: 0.11838177256830018
# ME: -15.171590062960217
# MAV: 515.473496128648
# MPV: 0.7399156949266682
# RME: -0.49893741627392274
# RMAE: 0.7399156949266682

############################# XGBoost ###########################

from xgboost import XGBRegressor # type: ignore

# Criar o modelo XGBoost
model = XGBRegressor(n_estimators=100, random_state=42)  # você pode ajustar o número de estimadores (árvores) conforme necessário

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
me = np.mean(y_test - y_pred)
mav = np.mean(np.abs(y_test))
mpv = np.mean(np.abs((y_test - y_pred) / (y_test + np.finfo(float).eps)))  # Adiciona um pequeno valor para evitar divisão por zero
rme = np.mean((y_test - y_pred) / (y_test + np.finfo(float).eps))  # Adiciona um pequeno valor para evitar divisão por zero
rmae = np.mean(np.abs(y_test - y_pred) / (np.abs(y_test) + np.finfo(float).eps))  # Adiciona um pequeno valor para evitar divisão por zero

print("Modelo: XGBoost")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# # Visualizar as previsões vs os valores reais
# plt.figure(figsize=(10, 5))
# plt.plot(y_test, label='Real')
# plt.plot(y_pred, label='Previsto')
# plt.legend()
# plt.show()

#### results

# MAE: 185.27285227909053
# MSE: 52071.94264778413
# RMSE: 228.19277518752457
# R2: 0.1572379469871521
# ME: -12.740157787965975
# MAV: 515.473496128648
# MPV: 0.7410286864461683
# RME: -0.4990561288246307
# RMAE: 0.7410286864461683



############################# LSTM ###########################

import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from sklearn.preprocessing import MinMaxScaler

# Normalização dos dados
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Redimensionar os dados para 3D (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Criar o modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Garantir que y_pred está no formato 1D
y_pred = y_pred.flatten()

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
me = np.mean(y_test - y_pred)
mav = np.mean(np.abs(y_test))
mpv = np.mean(np.abs((y_test - y_pred) / y_test))
rme = np.mean((y_test - y_pred) / y_test)
rmae = np.mean(np.abs(y_test - y_pred) / np.abs(y_test))

print("Modelo: LSTM")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# # Visualizar as previsões vs os valores reais
# plt.figure(figsize=(10, 5))
# plt.plot(y_test, label='Real')
# plt.plot(y_pred, label='Previsto')
# plt.legend()
# plt.show()

#### results

# MAE: 0.2451240157696407
# MSE: 0.08692460031842272
# RMSE: 0.29482978193938064
# R2: 0.13086141728497114
# ME: -0.018109347331303934
# MAV: 0.5782105548710534
# MPV: inf
# RME: -inf
# RMAE: inf