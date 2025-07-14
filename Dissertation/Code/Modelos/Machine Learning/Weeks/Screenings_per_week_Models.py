# Screenings_Count_per_week_Models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Carregar o DataFrame
df = pd.read_excel('C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Skin_clean and treated/Skin_clean.xlsx')

## Converter a data para datetime e ordenar de forma crescente
df['Screening_date'] = pd.to_datetime(df['Screening_date'])
df = df.sort_values(by='Screening_date')

# Iterar sobre cada data única no DataFrame
for date in df['Screening_date'].unique():
    # Filtrar o DataFrame para a data atual
    date_mask = df['Screening_date'] == date
    df_date = df[date_mask]
    
    # Preencher valores ausentes
    for column in df_date.columns:
        if column != 'Screening_date':  # Ignorar a coluna de data
            if pd.api.types.is_numeric_dtype(df_date[column]):
                # Verifica se a coluna tem valores não NaN antes de calcular a mediana
                if df_date[column].notna().any():
                    df.loc[date_mask, column] = df_date[column].fillna(df_date[column].median())
                else:
                    # Se todos os valores são NaN, preenche com a mediana global da coluna
                    df.loc[date_mask, column] = df_date[column].fillna(df[column].median())
            else:
                # Verifica se a coluna tem valores não NaN antes de calcular a moda
                if df_date[column].notna().any():
                    df.loc[date_mask, column] = df_date[column].fillna(df_date[column].mode().iloc[0])
                else:
                    # Se todos os valores são NaN, preenche com a moda global da coluna
                    if not df[column].mode().empty:
                        df.loc[date_mask, column] = df_date[column].fillna(df[column].mode().iloc[0])
                    else:
                        df.loc[date_mask, column] = df_date[column].fillna('Desconhecido')  # Ou outro valor padrão

print(df)

print(df.columns)

######## no groupping
# Adicionar uma coluna 'Week' que extrai o ano e semana do 'Screening_date'
df['Week'] = df['Screening_date'].dt.strftime('%Y-%U')

# Adicionar coluna com o número total de screenings por semana
df['Total_screenings'] = df.groupby('Week')['Screening_date'].transform('count')

# Mover a coluna 'Total_screenings' para a primeira posição
columns = ['Total_screenings'] + [col for col in df.columns if col != 'Total_screenings']
df = df[columns]

print(df[['Screening_date', 'Week', 'Total_screenings']])

####### groupping - uma linha por semana
# Definir as funções de agregação para cada coluna, mantendo 'Screening_date'
aggregations = {col: (lambda x: x.mode()[0] if x.dtype == 'O' else x.median()) for col in df.columns if col not in ['Screening_date', 'Week', 'Total_screenings']}
aggregations['Total_screenings'] = 'first'

# Agrupar os dados por semana, preservando a coluna 'Screening_date'
df = df.groupby(['Week', 'Screening_date']).agg(aggregations).reset_index()

# Exibir o DataFrame resultante
print(df)

print(df.columns)

# Eliminar a coluna 'Week'
df = df.drop(columns=['Week'])

print(df.columns)

# Definir as features (X) e a variável target (y)
X = df.drop(columns=['Total_screenings', 'Screening_date'])
y = df['Total_screenings']

print(X.columns)
print(y.head(75))

# # Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Definindo as datas de corte
# split_date = '2024-03-01'

# # Dividindo os dados com base na data
# train = df[df['Screening_date'] < split_date]
# test = df[df['Screening_date'] >= split_date]

# # Separando X e y para cada conjunto
# X_train = train.drop(columns=['Total_screenings', 'Screening_date'])
# y_train = train['Total_screenings']

# X_test = test.drop(columns=['Total_screenings', 'Screening_date'])
# y_test = test['Total_screenings']

# # print(X_train)

# ####################### MLR ######################

from sklearn.linear_model import LinearRegression

# Adicionar uma constante aos dados
X_train_sm = sm.add_constant(X_train)
X_test_sm = X_test.copy()  # Certifique-se de que não está sobrescrevendo X_test original
X_test_sm.insert(0, 'const', 1.0)

print(X_train_sm.head())
print(X_test_sm.head())

# Ajustar o modelo
model_sm = sm.OLS(y_train, X_train_sm).fit()

# # Obter o resumo do modelo
# print(model_sm.summary())

# Fazer previsões no conjunto de teste
y_pred = model_sm.predict(X_test_sm)

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

# print("Modelo: LR")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')


#### results

# Modelo: LR with shuffle
# MAE: 69.03493044201907
# MSE: 7480.655089559562
# RMSE: 86.49078037316788
# R2: 0.13215500728572704
# ME: 36.87552677621236
# MAV: 167.83333333333334
# MPV: 1.3452684769380712
# RME: -0.8923945924679075
# RMAE: 1.3452684769380712

# Modelo: LR with split date
# MAE: 81.81730531050039
# MSE: 9994.175369256973
# RMSE: 99.97087260425896
# R2: -0.6786684620385823
# ME: 75.13033919278999
# MAV: 226.28571428571428
# MPV: 0.3327331284727909
# RME: 0.29639092131132144
# RMAE: 0.3327331284727909


####################### GLM ######################

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Gaussian
from sklearn.preprocessing import PolynomialFeatures

### Gaussian ###

# Ajustar o modelo GLM com família Gaussian
model_glm = GLM(y_train, X_train_sm, family=Gaussian()).fit()

# # Obter o resumo do modelo
# print(model_glm.summary())

# Fazer previsões no conjunto de teste
y_pred = model_glm.predict(X_test_sm)

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

print("Modelo: GLM com Gaussian")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

#### results

# Modelo: GLM com Gaussian shuffle
# MAE: 69.03493044201907
# MSE: 7480.655089559562
# RMSE: 86.49078037316788
# R2: 0.13215500728572704
# ME: 36.87552677621236
# MAV: 167.83333333333334
# MPV: 1.3452684769380712
# RME: -0.8923945924679075
# RMAE: 1.3452684769380712

# Modelo: GLM com Gaussian split date
# MMAE: 81.81730531050039
# MSE: 9994.175369256973
# RMSE: 99.97087260425896
# R2: -0.6786684620385823
# ME: 75.13033919278999
# MAV: 226.28571428571428
# MPV: 0.3327331284727909
# RME: 0.29639092131132144
# RMAE: 0.3327331284727909


####### Kernel Polinomial #######

# Definir o grau do polinômio
degree = 2

# Criar o transformador polinomial
poly = PolynomialFeatures(degree)

# Ajustar e transformar os dados de treino
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Adicionar uma constante aos dados transformados
X_train_poly_sm = sm.add_constant(X_train_poly)
X_test_poly_sm = sm.add_constant(X_test_poly)

# print(X_train_poly_sm)
# print(X_test_poly_sm)

# Ajustar o modelo GLM com família Gaussian
model_glm = GLM(y_train, X_train_poly_sm, family=Gaussian()).fit()

# # Obter o resumo do modelo
# print(model_glm.summary())

# Fazer previsões no conjunto de teste
y_pred = model_glm.predict(X_test_poly_sm)

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

# print("Modelo: GLM com Kernel Polinomial")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')


#### results

# Modelo: GLM com Kernel Polinomial with no shuffle
# MAE: 110.75345149607912
# MSE: 21080.224285137076
# RMSE: 145.190303688425
# R2: -1.445556822514476
# ME: 73.76441725577641
# MAV: 167.83333333333334
# MPV: 1.543485849841001
# RME: -0.6992681230567408
# RMAE: 1.543485849841001

# Modelo: GLM com Kernel Polinomial with split date
# MAE: 122.46342663034739
# MSE: 24539.25076965165
# RMSE: 156.65009023186565
# R2: -3.121727388913408
# ME: 109.27931952379458
# MAV: 226.28571428571428
# MPV: 0.498617180249851
# RME: 0.4201403522346559
# RMAE: 0.498617180249851


####################### SVR KERNEL GAUSSIANO ######################

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Normalizar a coluna 'Age'
scaler = StandardScaler()
X_train['Age'] = scaler.fit_transform(X_train[['Age']])
X_test['Age'] = scaler.fit_transform(X_test[['Age']])

# Definir o modelo SVR com kernel RBF
model_svr = SVR(kernel='rbf', C=1.0, gamma='scale')

# Ajustar o modelo
model_svr.fit(X_train, y_train)

# # "Summary" of the model
# print(f"Support Vectors: {model_svr.support_vectors_}")
# print(f"Number of Support Vectors: {model_svr.n_support_}")
# print(f"Dual Coefficients: {model_svr.dual_coef_}")

# Fazer previsões no conjunto de teste
y_pred = model_svr.predict(X_test)

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

# print("Modelo: SVR")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

#### results

# Modelo: SVR kernel gaussiano with shuffle
# MAE: 96.39734400652065
# MSE: 14345.580051216602
# RMSE: 119.77303557652951
# R2: -0.6642579648412976
# ME: 75.44053163301949
# MAV: 167.83333333333334
# MPV: 1.341604204646704
# RME: -0.585526894654433
# RMAE: 1.341604204646704

# Modelo: SVR kernel gaussiano with shuffle
# MAE: 129.9011231414846
# MSE: 22813.55551112784
# RMSE: 151.0415688184145
# R2: -2.8318715380260517
# ME: 129.9011231414846
# MAV: 226.28571428571428
# MPV: 0.5394686543841899
# RME: 0.5394686543841899
# RMAE: 0.5394686543841899

####################### SVR LINEAR KERNEL ######################

# Definir o modelo SVR com kernel RBF
model_svr_linear = SVR(kernel='linear', C=1.0)

# Ajustar o modelo
model_svr_linear.fit(X_train, y_train)

# "Summary" of the model
# print(f"Support Vectors: {model_svr_linear.support_vectors_}")
# print(f"Number of Support Vectors: {model_svr_linear.n_support_}")
# print(f"Dual Coefficients: {model_svr_linear.dual_coef_}")

# Fazer previsões no conjunto de teste
y_pred = model_svr_linear.predict(X_test)

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

# print("Modelo: SVR linear")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

#### results
# Modelo: SVR linear kernel with shuffle
# MAE: 129.10026526655093
# MSE: 22429.18847413157
# RMSE: 149.76377557384018
# R2: -2.7673114518745088
# ME: 129.10026526655093
# MAV: 226.28571428571428
# MPV: 0.5373501298201109
# RME: 0.5373501298201109
# RMAE: 0.5373501298201109

# Modelo: SVR linear kernel with split date
# MAE: 96.65549801985392
# MSE: 14343.242222377115
# RMSE: 119.76327576672708
# R2: -0.6639867488812137
# ME: 74.84830623942076
# MAV: 167.83333333333334
# MPV: 1.3869330278704817
# RME: -0.6323799380527588
# RMAE: 1.3869330278704817

####################### MLP NN: Multilayer Perceptron Neural Network ######################

from sklearn.neural_network import MLPRegressor

# Definir o modelo MLP com uma camada oculta de 100 neurônios (você pode ajustar os parâmetros conforme necessário)
model_mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Ajustar o modelo aos dados de treino
model_mlp.fit(X_train, y_train)

# "Summary" do modelo
# print(f"Coefs: {model_mlp.coefs_}")
# print(f"Intercepts: {model_mlp.intercepts_}")
# print(f"Número de iterações: {model_mlp.n_iter_}")

# Fazer previsões no conjunto de teste
y_pred = model_mlp.predict(X_test)

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

# print("Modelo: MPL NN")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

#### results

# Modelo: MPL NN with shuffle
# MAE: 83.65533562021143
# MSE: 10052.457311872813
# RMSE: 100.26194348741107
# R2: -0.16620464894290987
# ME: 47.81576020620951
# MAV: 167.83333333333334
# MPV: 1.5821536505704923
# RME: -1.00067759036575
# RMAE: 1.5821536505704923

# Modelo: MPL NN with split date
# MAE: 108.45525526044176
# MSE: 16376.234240416026
# RMSE: 127.96966140619435
# R2: -1.7506289344196824
# ME: 108.45525526044176
# MAV: 226.28571428571428
# MPV: 0.4512055291673504
# RME: 0.4512055291673504
# RMAE: 0.4512055291673504

####################### LSTM ######################

# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import LSTM, Dense # type: ignore
# from tensorflow.keras.optimizers import Adam # type: ignore

# # Certifique-se de que seus dados estejam no formato adequado para o LSTM
# # LSTM espera os dados na forma de [samples, time_steps, features]
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# # Definir o modelo LSTM
# model_lstm = Sequential()
# model_lstm.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
# model_lstm.add(Dense(1))  # Camada de saída

# # Compilar o modelo
# model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# # Ajustar o modelo aos dados de treino
# model_lstm.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# # "Summary" do modelo
# model_lstm.summary()

# # Fazer previsões no conjunto de teste
# y_pred = model_lstm.predict(X_test)

# print(f"Shape of y_test: {y_test.shape}")
# print(f"Shape of y_pred: {y_pred.shape}")

# # Flatten y_pred to make it a 1D array
# y_pred = y_pred.flatten()

# # Calcular e imprimir as métricas de avaliação
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
# me = np.mean(y_test - y_pred)
# mav = np.mean(np.abs(y_test))
# mpv = np.mean(np.abs((y_test - y_pred) / y_test))
# rme = np.mean((y_test - y_pred) / y_test)
# rmae = np.mean(np.abs(y_test - y_pred) / np.abs(y_test))

# print("Modelo: MPL NN")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

#### results

# LSTM split date
# MAE: 200.90199879237585
# MSE: 46315.256469999804
# RMSE: 215.20979640806272
# R2: -6.779328346252441
# ME: 200.90199879237585
# MAV: 226.28571428571428
# MPV: 0.8787100521710753
# RME: 0.8787100521710753
# RMAE: 0.8787100521710753

# LSTM shuffle
# MAE: 145.90236155192056
# MSE: 29330.18006079944
# RMSE: 171.2605618956082
# R2: -2.4026503562927246
# ME: 143.9110247294108
# MAV: 167.83333333333334
# MPV: 0.9246246341770118
# RME: 0.5927351637587175
# RMAE: 0.924624634177011


####################### XGBOOST ######################

from xgboost import XGBRegressor

# Definir o modelo XGBoost
model_xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Ajustar o modelo aos dados de treino
model_xgb.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model_xgb.predict(X_test)

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

# print("Modelo: XGB")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

#### results

# XGB shuffle
# MAE: 86.87526925404866
# MSE: 14025.612822753137
# RMSE: 118.42978013469896
# R2: -0.6271380186080933
# ME: 35.84446493784586
# MAV: 167.83333333333334
# MPV: 0.9561680008484085
# RME: -0.42061832849765257
# RMAE: 0.9561680008484085

# XGB split date
# MAE: 102.23842348371234
# MSE: 18053.9411328567
# RMSE: 134.36495500262225
# R2: -2.0324244499206543
# ME: 100.00316783360073
# MAV: 226.28571428571428
# MPV: 0.4039733715175205
# RME: 0.39268420156746187
# RMAE: 0.4039733715175205


####################### CNN ######################

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# # Dividir os dados em conjuntos de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Definindo as datas de corte
split_date = '2024-01-01'

# Dividindo os dados com base na data
train = df[df['Screening_date'] < split_date]
test = df[df['Screening_date'] >= split_date]

# Separando X e y para cada conjunto
X_train = train.drop(columns=['Total_screenings', 'Screening_date'])
y_train = train['Total_screenings']

X_test = test.drop(columns=['Total_screenings', 'Screening_date'])
y_test = test['Total_screenings']

print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de y_test: {y_test.shape}")

def create_sequences(data, labels, window_size):
    sequences = []
    sequence_labels = []
    
    for i in range(len(data) - window_size):
        seq = data.iloc[i:i+window_size].values
        label = labels.iloc[i+window_size]
        sequences.append(seq)
        sequence_labels.append(label)
    
    return np.array(sequences), np.array(sequence_labels)

# Tamanho da janela (número de passos de tempo a considerar)
window_size = 5

# Reformular X_train e y_train
X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, window_size)

print(f"Shape de X_train_seq: {X_train_seq.shape}")
print(f"Shape de X_test_seq: {X_test_seq.shape}")
print(f"Shape de y_train_seq: {y_train_seq.shape}")
print(f"Shape de y_test_seq: {y_test_seq.shape}")

model_cnn = Sequential()

# Camada convolucional 1D
model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))

# Camada de pooling 1D
model_cnn.add(MaxPooling1D(pool_size=2))

# Achatar a saída e adicionar camadas densas
model_cnn.add(Flatten())
model_cnn.add(Dense(50, activation='relu'))
model_cnn.add(Dropout(0.5))

# Camada de saída
model_cnn.add(Dense(1, activation='linear'))

# Compilar o modelo
model_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Treinar o modelo
model_cnn.fit(X_train_seq, y_train_seq, epochs=100, batch_size=32, validation_data=(X_test_seq, y_test_seq), verbose=1)

# Fazer previsões no conjunto de teste
y_pred = model_cnn.predict(X_test_seq)

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test_seq, y_pred)
mse = mean_squared_error(y_test_seq, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_seq, y_pred)
me = np.mean(y_test_seq - y_pred)
mav = np.mean(np.abs(y_test_seq))
mpv = np.mean(np.abs((y_test_seq - y_pred) / y_test_seq))
rme = np.mean((y_test_seq - y_pred) / y_test_seq)
rmae = np.mean(np.abs(y_test_seq - y_pred) / np.abs(y_test_seq))

# print("Modelo: CNN")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

#### results

# Modelo: CNN shuffle
# MAE: 96.035888671875
# MSE: 13415.935413196405
# RMSE: 115.82717907812659
# R2: -1.008159875869751
# ME: 82.37169940655048
# MAV: 199.46153846153845
# MPV: 0.4666217102718543
# RME: 0.27845845171435796
# RMAE: 0.4666217102718543

# Modelo: CNN split date
# MAE: 100.17866092258029
# MSE: 15111.604303281878
# RMSE: 122.92926544676772
# R2: -1.2301819324493408
# ME: 90.94761064317491
# MAV: 206.77777777777777
# MPV: 0.44763165531946036
# RME: 0.33985340137472286
# RMAE: 0.44763165531946036


