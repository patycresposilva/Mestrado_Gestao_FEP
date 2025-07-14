# date from 1 to 90

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

# Agrupar os dados por dias, usando mediana para numéricas e moda para categóricas
df = df.groupby('Screening_date').agg(lambda x: x.mode()[0] if x.dtype == 'O' else x.median()).reset_index()

print(df)

# Número total de datas diferentes
num_dates = df['Screening_date'].nunique()
print(f'Total de datas diferentes: {num_dates}')

# Criar uma nova coluna numérica contínua baseada nas datas
df['Date'] = range(1, num_dates + 1)

# Reordenar as colunas para colocar a nova coluna 'Date' em primeiro lugar
cols = ['Date'] + [col for col in df.columns if col != 'Date']
df = df[cols]

print(df)

print(df.columns)

# Definir as features (X) e a variável target (y)
X = df.drop(columns=['Screening_date', 'Date'])
y = df['Date']

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

print("Modelo: Linear Regression")
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

# MAE: 22.487137084972744
# MSE: 754.8074025844621
# RMSE: 27.473758435723028
# R2: -0.1787442146080005
# ME: -8.482512932230335
# MAV: 39.388888888888886
# MPV: 3.896842464524595
# RME: -3.710254812129865
# RMAE: 3.896842464524595

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

# MAE: 26.866666666666664
# MSE: 1038.2044444444446
# RMSE: 32.22118005977504
# R2: -0.621310917565177
# ME: -8.711111111111112
# MAV: 39.388888888888886
# MPV: 4.738690303687758
# RME: -4.489388989998731
# RMAE: 4.738690303687758


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

# # Visualizar as previsões vs os valores reais
# plt.figure(figsize=(10, 5))
# plt.plot(y_test, label='Real')
# plt.plot(y_pred, label='Previsto')
# plt.legend()
# plt.show()

#### results

# MAE: 24.785397733439396
# MSE: 945.3374530782726
# RMSE: 30.746340482702532
# R2: -0.4762852746977211
# ME: -5.7661669395419395
# MAV: 39.388888888888886
# MPV: 4.87945116334008
# RME: -4.595516778131706
# RMAE: 4.87945116334008

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

# MAE: 27.960174560546875
# MSE: 1185.170531451762
# RMSE: 34.42630580605131
# R2: -0.8508205413818359
# ME: -10.876426272922092
# MAV: 39.388888888888886
# MPV: 5.803591134580069
# RME: -5.549921468320606
# RMAE: 5.803591134580069


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

# MAE: 0.2374029062437208
# MSE: 0.08127381204097689
# RMSE: 0.28508562229789297
# R2: -0.005342556945777455
# ME: -0.07415263501743552
# MAV: 0.43133583021223465
# MPV: inf
# RME: -inf
# RMAE: inf



