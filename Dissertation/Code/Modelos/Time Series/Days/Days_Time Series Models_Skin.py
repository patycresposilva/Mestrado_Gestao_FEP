################ TIME SERIES MODELS ################

import warnings
warnings.filterwarnings("ignore")


############################### AUTOREGRESSIVE ####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import pacf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import probplot
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid

# Load your data
data = pd.read_excel('C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Merged_File_v7_skin.xlsx')

# Criar uma nova coluna para preservar os valores de 'Screening_date'
data['Screening_date_original'] = data['Screening_date']

# Converter a coluna 'Screening_date' para datetime
data['Screening_date'] = pd.to_datetime(data['Screening_date'])

# Agrupar por data e somar o número de rastreios realizados em cada data
data_aggregated = data.groupby('Screening_date').size()

# Verificar os nomes das colunas para garantir que estamos usando a coluna correta
print(data_aggregated.head())

# Transformar a série agregada em um DataFrame e definir a frequência do índice
data_aggregated = data_aggregated.to_frame(name='count')
data_aggregated = data_aggregated.asfreq('D').fillna(0)

# Definir a coluna 'Screening_date' como índice
data.set_index('Screening_date', inplace=True)

# Ordenar o índice cronologicamente
data = data.sort_index()

# Print the column names to verify
print(data.columns)

# Definir a variável alvo
y = data_aggregated['count']

print(y)

# Verificar se o índice ainda é datetime
print("Index type:", type(y.index))

# Check for stationarity
result = adfuller(y)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# quando usávamos as datas sem agrupar dava isto:
# # ADF Statistic: 7.151302662253552
# # p-value: 1.0
# # series is non-stationary
# mas agora, com as datas agrupadas a série já fica estacionária,
# não havendo necessidade de diferenciar
# ADF Statistic: -2.996532692807403
# p-value: 0.03521947433465731

# If the series is not stationary, difference it
# y_diff = y.diff().dropna()

# # Verificar se o índice foi corrigido
# print("Index type after fix:", type(y_diff.index))

# # Verificar a estacionariedade após a diferenciação
# result_diff = adfuller(y_diff)
# print('ADF Statistic after differencing:', result_diff[0])
# print('p-value after differencing:', result_diff[1])

# ADF Statistic after differencing: -50.68944583464653
# p-value after differencing: 0.0
# sugere uma forte evidência contra a hipótese nula de que a série possui uma raiz unitária
# hipótese nula pode ser rejeitada com um alto nível de confiança

# Plotar a PACF para identificar a ordem do modelo AR
plot_pacf(y, lags=40)
plt.show()

# OU ##
# Calcular o PACF
pacf_values, confint = pacf(y, nlags=40, alpha=0.05)

# Determinar os lags significativos
significant_lags = np.where((confint[:, 0] > 0) | (confint[:, 1] < 0))[0]

# Print the significant lags
print("Significant lags:", significant_lags)

# antes, nós diferenciámos a vv para ficar estacionária e os lags importantes eram só o 1

# O gráfico PACF sugere que o modelo AR(1) é apropriado, 
# pois a autocorrelação parcial é significativa apenas no lag 1
# Num modelo AR(1), usamos apenas o valor anterior (lag1)
# para prever o valor atual.

# From the PACF plot, it appears that the time series is best described by
# an AR model of order 1 (AR(1)). This means the current value of the
# series can be sufficiently explained by the immediately preceding value.


# AGORA, os lags importantes são: Significant lags: [ 0  7 14]

# # Verificar se as datas de divisão estão presentes no índice
# print("Min date:", y.index.min())
# print("Max date:", y.index.max())

# Dividir o conjunto de dados em treino e teste baseado na data
train = y[:'2024-02-29']  # Dados até 29-02-2024, exclusivo
test = y['2024-03-01':]   # Dados a partir de 01-03-2024

# # 1. Dividir os dados em treino e teste (80% treino, 20% teste)
# train_size = int(len(data_aggregated) * 0.8)
# train = data_aggregated['count'][:train_size]
# test = data_aggregated['count'][train_size:]

# Ajustar o modelo AR(7, 14) nos dados
model1 = AutoReg(train, lags=[7, 14, 21, 28]).fit()
print(model1.summary())

# Fazer previsões
predictions = model1.predict(start=test.index[0], end=test.index[-1], dynamic=False)

# Avaliar o modelo
mae = mean_absolute_error(test, predictions)
mse = mean_squared_error(test, predictions)
r2 = r2_score(test, predictions)
print("Modelo AR")
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')


# # Plotar os resultados
# plt.figure(figsize=(10, 6))
# plt.plot(train.index, train, label='Treino')
# plt.plot(test.index, test, label='Teste')
# plt.plot(test.index, predictions, label='Previsões', color='red')
# plt.legend()
# plt.show()

# Modelo AR (7,14)
# Mean Absolute Error: 30.957209088366163
# Mean Squared Error: 5164.539198146904
# R-squared: 0.16059619631383448

# Modelo AR (7,14,21,28)
# Mean Absolute Error: 25.39082950211918
# Mean Squared Error: 3565.143366397804
# R-squared: 0.42054948416025284

# #CONCLUSION##
# este modelo não foi capaz de captar totalmente a variablidade dos dados, 
# mas demonstrou que time series é promissor
# neste sentido, modelos time series mais complexos que permitam dados adicionais
# podem ajudar a melhorar estas previsões

# ################################ ARMA ##############################################
#captura lags consecutivos, enquando AR dá para especificar!

# Ajustar o modelo ARMA nos dados
# Use grid search or specify your own (p, q) values
# p_values = range(0, 7)
# q_values = range(0, 4)
# aic_values = []

# for p in p_values:
#     for q in q_values:
#         try:
#             model = ARIMA(train, order=(p, 0, q))
#             model_fit = model.fit()
#             aic_values.append((p, q, model_fit.aic))
#         except:
#             continue

# # Escolher os melhores valores (p, q) com base no AIC
# best_p, best_q, best_aic = min(aic_values, key=lambda x: x[2])
# print(f'Best p: {best_p}, Best q: {best_q}, Best AIC: {best_aic}')

# # Ajustar o melhor modelo ARMA
# best_model_ARMA = ARIMA(train, order=(best_p, 0, best_q)).fit()
# print(best_model_ARMA.summary())

# # Fazer previsões
# predictions = best_model_ARMA.predict(start=test.index[0], end=test.index[-1], dynamic=False)

# # Avaliar o modelo
# mae = mean_absolute_error(test, predictions)
# mse = mean_squared_error(test, predictions)
# r2 = r2_score(test, predictions)
# print("Modelo ARMA")
# print(f'Mean Absolute Error: {mae}')
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')

# # Plotar os resultados
# plt.figure(figsize=(10, 6))
# plt.plot(train.index, train, label='Treino')
# plt.plot(test.index, test, label='Teste')
# plt.plot(test.index, predictions, label='Previsões', color='red')
# plt.legend()
# plt.show()

# # Modelo ARMA (p=4)
# # Mean Absolute Error: 36.892887957168625
# # Mean Squared Error: 5311.57625395863
# # R-squared: 0.13669794340184882

# ARIMA (p=7)
# Mean Absolute Error: 35.32252264199114
# Mean Squared Error: 5540.00864645892
# R-squared: 0.09957032914757513

# # Plotar resíduos
# residuals = best_model_ARMA.resid
# plt.figure(figsize=(10, 6))
# plt.subplot(211)
# plt.plot(residuals)
# plt.title('Residuals')
# plt.subplot(212)
# plot_acf(residuals, lags=20, ax=plt.gca())
# plt.show()

# # check normality of residuals
# from scipy.stats import probplot

# plt.figure(figsize=(10, 6))
# probplot(residuals, dist="norm", plot=plt)
# plt.title('Q-Q Plot of Residuals')
# plt.show()


# ##################################### SARIMA ##########################################
#captura lags consecutivos, enquando AR dá para especificar!
# adds seasonality (S) to the data

# # Ajustar o modelo SARIMA (p=1, d=0, q=1) (P=1, D=0, Q=1, S=7) como exemplo para sazonalidade semanal
# model_sarima = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)).fit()
# print(model_sarima.summary())

# # Fazer previsões
# predictions_sarima = model_sarima.predict(start=test.index[0], end=test.index[-1])

# # Avaliar o modelo
# mae = mean_absolute_error(test, predictions_sarima)
# mse = mean_squared_error(test, predictions_sarima)
# r2 = r2_score(test, predictions_sarima)
# print("Modelo SARIMA")
# print(f'Mean Absolute Error: {mae}')
# print(f'Mean Squared Error: {mse}')
# print(f'R-squared: {r2}')

# plt.figure(figsize=(14, 7))
# plt.plot(train.index, train, label='Train')
# plt.plot(test.index, test, label='Test')
# plt.plot(predictions_sarima.index, predictions_sarima, label='SARIMA Predictions', color='red')
# plt.xlabel('Screening_date')
# plt.ylabel('count of Screenings')
# plt.title('SARIMA Model - Actual vs Predicted')
# plt.legend()
# plt.show()

# SARIMA order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)
# Mean Absolute Error: 17.937484849621136
# Mean Squared Error: 2670.5329241073787
# R-squared: 0.5659524677111167

# SARIMA order=(4, 0, 1), seasonal_order=(4, 0, 1, 7)
# Mean Absolute Error: 20.924865937998177
# Mean Squared Error: 3581.602637532114
# R-squared: 0.4178743229762698

# adressing the warning Non-stationary:
# starting seasonal autoregressive Using zeros as starting parameters
# it is stationairy but there are some aspects that can be improved to strengthen the stationarity

# # Fit SARIMA with seasonal differencing
# model_sarima_seasonal_diff = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 7)).fit()
# print(model_sarima_seasonal_diff.summary())

# # Predictions
# predictions_sarima_seasonal_diff = model_sarima_seasonal_diff.predict(start=test.index[0], end=test.index[-1])

# # Evaluation
# mae_sarima_seasonal_diff = mean_absolute_error(test, predictions_sarima_seasonal_diff)
# mse_sarima_seasonal_diff = mean_squared_error(test, predictions_sarima_seasonal_diff)
# r2 = r2_score(test, predictions_sarima)
# print("Modelo SARIMA SEO DIFF")
# print(f'Mean Absolute Error (SARIMA with seasonal differencing): {mae_sarima_seasonal_diff}')
# print(f'Mean Squared Error (SARIMA with seasonal differencing): {mse_sarima_seasonal_diff}')
# print(f'R-squared: {r2}')

# plt.figure(figsize=(14, 7))
# plt.plot(train.index, train, label='Train')
# plt.plot(test.index, test, label='Test')
# plt.plot(predictions_sarima.index, predictions_sarima, label='SARIMA Predictions', color='red')
# plt.plot(predictions_sarima_seasonal_diff.index, predictions_sarima_seasonal_diff, label='SARIMA Predictions with Seasonal Differencing', color='green')
# plt.xlabel('Date')
# plt.ylabel('count of Screenings')
# plt.title('SARIMA Models - Actual vs Predicted')
# plt.legend()
# plt.show()

# Modelo SARIMA SEO DIFF order=(1, 0, 1), seasonal_order=(1, 1, 1, 7)
# Mean Absolute Error (SARIMA with seasonal differencing): 17.432775312380272
# Mean Squared Error (SARIMA with seasonal differencing): 2555.531797342903
# R-squared: 0.5659524677111167

# Modelo SARIMA SEO DIFF order=(4, 0, 1), seasonal_order=(4, 1, 1, 7)
# Mean Absolute Error (SARIMA with seasonal differencing): 22.2887384623647
# Mean Squared Error (SARIMA with seasonal differencing): 3521.3097089247076
# R-squared: 0.4178743229762698


# ########### grid search + increase number of iterations to 1000 + seasonal differencing ############

# # in SARIMA, beyond the three parameters from ARIMA, we have 4 more:
# # P: Number of seasonal autoregressive terms
# # D: Number of seasonal differentiations
# # Q: Number of seasonal moving average terms
# # s: Seasonality period (e.g. s=12 for monthly data with annual seasonality, s=7 for daily data with weekly seasonality)

# # apply seasonal differencing
# train_diff = train.diff(7).dropna()
# test_diff = test.diff(7).dropna()

# # Definir o intervalo de parâmetros para busca de grade
# p = range(0, 4)
# d = [0]
# q = range(0, 1)
# P = range(0, 4)
# D = range(0, 2)
# Q = range(0, 2)
# s = [7]  # Período sazonal semanal

# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = list(itertools.product(P, D, Q, s))

# best_aic = np.inf
# best_params = None

# for param in pdq:
#     for seasonal_param in seasonal_pdq:
#         try:
#             model = SARIMAX(train_diff, order=param, seasonal_order=seasonal_param).fit(disp=False, maxiter=1000)
#             if model.aic < best_aic:
#                 best_aic = model.aic
#                 best_params = (param, seasonal_param)
#         except:
#             continue

# print(f'Best SARIMA parameters: {best_params}')

# BEST SARIMA PARAMETERS
# Best SARIMA parameters: ((0, 0, 0), (0, 0, 1, 7))



############## BEST MODEL #############

# # Fit SARIMA with seasonal differencing
# model_sarima_seasonal_diff = SARIMAX(train, order=(1, 0, 0), seasonal_order=(2, 2, 2, 7)).fit()
# print(model_sarima_seasonal_diff.summary())

# # Predictions
# predictions_sarima_seasonal_diff = model_sarima_seasonal_diff.predict(start=test.index[0], end=test.index[-1])

# # Evaluation
# mae_sarima_seasonal_diff = mean_absolute_error(test, predictions_sarima_seasonal_diff)
# mse_sarima_seasonal_diff = mean_squared_error(test, predictions_sarima_seasonal_diff)
# r2 = r2_score(test, predictions_sarima)
# print("Modelo SARIMA SEO DIFF")
# print(f'Mean Absolute Error (SARIMA with seasonal differencing): {mae_sarima_seasonal_diff}')
# print(f'Mean Squared Error (SARIMA with seasonal differencing): {mse_sarima_seasonal_diff}')
# print(f'R-squared: {r2}')

# plt.figure(figsize=(14, 7))
# plt.plot(train.index, train, label='Train')
# plt.plot(test.index, test, label='Test')
# plt.plot(predictions_sarima.index, predictions_sarima, label='SARIMA Predictions', color='red')
# plt.plot(predictions_sarima_seasonal_diff.index, predictions_sarima_seasonal_diff, label='SARIMA Predictions with Seasonal Differencing', color='green')
# plt.xlabel('Date')
# plt.ylabel('count of Screenings')
# plt.title('SARIMA Models - Actual vs Predicted')
# plt.legend()
# plt.show()

# Modelo SARIMA SEO DIFF BEST PARAMS (1,0,0) (2,2,2,7)
# Mean Absolute Error (SARIMA with seasonal differencing): 14.631282808763876
# Mean Squared Error (SARIMA with seasonal differencing): 2202.921353744248
# R-squared: 0.5659524677111167

# Modelo SARIMA SEO DIFF order=(0, 0, 0), seasonal_order=(0, 0, 1, 7)
# Mean Absolute Error (SARIMA with seasonal differencing): 26.20475172141122
# Mean Squared Error (SARIMA with seasonal differencing): 6531.563383325058
# R-squared: 0.5659524677111167


############################## EXPONENTIAL SMOOTHING ####################################

##### simple ######
# PARA SÉRIES SEM TENDÊNCIA E SEM SAZONALIDADE

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# # 2. Criar e ajustar o modelo SES
# model = SimpleExpSmoothing(train).fit()

# # 3. Fazer previsões
# y_pred = model.predict(start=len(train), end=len(train) + len(test) - 1)

# # 4. Avaliar o modelo
# mae = mean_absolute_error(test, y_pred)
# mse = mean_squared_error(test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(test, y_pred)
# me = np.mean(test - y_pred)
# mav = np.mean(np.abs(test))
# mpv = np.mean(np.abs((test - y_pred) / test))
# rme = np.mean((test - y_pred) / test)
# rmae = np.mean(np.abs(test - y_pred) / np.abs(test))

# print("Modelo: SES")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')


# plt.figure(figsize=(12, 6))
# plt.plot(train.index, train, label='Dados de Treino', color='blue')
# plt.plot(test.index, test, label='Dados Reais de Teste', color='green')
# plt.plot(test.index, y_pred, label='Previsão SES', color='red', linestyle='--')
# plt.xlabel('Data')
# plt.ylabel('Contagem')
# plt.title('Previsão SES vs. Dados Reais')
# plt.legend()
# plt.grid(True)
# plt.show()

# Modelo: SES
# MAE: 41.88729193501876
# MSE: 6218.1613529272745
# RMSE: 78.85531911626047
# R2: -0.010651307178441316
# ME: 8.095278185970571
# MAV: 27.310344827586206
# MPV: inf
# RME: -inf
# RMAE: inf

##### double ######
# O modelo é capaz de capturar tendências lineares, mas não considera sazonalidade.

# from statsmodels.tsa.holtwinters import ExponentialSmoothing

# # 2. Criar e ajustar o modelo Holt's Linear Trend (Double Exponential Smoothing)
# model = ExponentialSmoothing(train, trend='add').fit()

# # 3. Fazer previsões
# y_pred = model.predict(start=len(train), end=len(train) + len(test) - 1)

# # 4. Avaliar o modelo
# mae = mean_absolute_error(test, y_pred)
# mse = mean_squared_error(test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(test, y_pred)
# me = np.mean(test - y_pred)
# mav = np.mean(np.abs(test))
# mpv = np.mean(np.abs((test - y_pred) / test))
# rme = np.mean((test - y_pred) / test)
# rmae = np.mean(np.abs(test - y_pred) / np.abs(test))

# print("Modelo: HLTM")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

# plt.figure(figsize=(12, 6))
# plt.plot(train.index, train, label='Dados de Treino', color='blue')
# plt.plot(test.index, test, label='Dados Reais de Teste', color='green')
# plt.plot(test.index, y_pred, label='Previsão HWLT', color='red', linestyle='--')
# plt.xlabel('Data')
# plt.ylabel('Contagem')
# plt.title('Previsão HWLT vs. Dados Reais')
# plt.legend()
# plt.grid(True)
# plt.show()

# Modelo: HLTM 
# MAE: 33.5840977287367
# MSE: 6646.071044533921
# RMSE: 81.52343862064407
# R2: -0.08020040129659112
# ME: 21.2272539509069
# MAV: 27.310344827586206
# MPV: inf
# RME: nan
# RMAE: inf

##### triple ######
# lidar com séries temporais que têm tanto tendência quanto sazonalidade.

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 2. Criar e ajustar o modelo Holt-Winters (Triple Exponential Smoothing)
# Definindo sazonalidade com período de 52 semanas (ajuste conforme necessário)
model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()

# 3. Fazer previsões
y_pred = model.predict(start=len(train), end=len(train) + len(test) - 1)

# 4. Avaliar o modelo
mae = mean_absolute_error(test, y_pred)
mse = mean_squared_error(test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(test, y_pred)
me = np.mean(test - y_pred)
mav = np.mean(np.abs(test))
mpv = np.mean(np.abs((test - y_pred) / test))
rme = np.mean((test - y_pred) / test)
rmae = np.mean(np.abs(test - y_pred) / np.abs(test))

print("Modelo: HWES")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# plt.figure(figsize=(12, 6))
# plt.plot(train.index, train, label='Dados de Treino', color='blue')
# plt.plot(test.index, test, label='Dados Reais de Teste', color='green')
# plt.plot(test.index, y_pred, label='Previsão HWES', color='red', linestyle='--')
# plt.xlabel('Data')
# plt.ylabel('Contagem')
# plt.title('Previsão HWES vs. Dados Reais')
# plt.legend()
# plt.grid(True)
# plt.show()

#  Modelo: HWES seasonal_periods=4
# MAE: 47.31291813857395
# MSE: 6162.899334028711
# RMSE: 78.50413577658638
# R2: -0.0016694508921193396
# ME: 0.7916151805477325
# MAV: 27.310344827586206
# MPV: inf
# RME: -inf
# RMAE: inf

#  Modelo: HWES seasonal_periods=7
# Modelo: HWES
# MAE: 20.763809496049102
# MSE: 2484.1842197790616
# RMSE: 49.84159126451584
# R2: 0.5962401284730496
# ME: 3.118550415907782
# MAV: 27.310344827586206
# MPV: inf
# RME: -inf
# RMAE: inf