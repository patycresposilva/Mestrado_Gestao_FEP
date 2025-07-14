# Regression_Date_as_reference

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

# Ver os valores únicos da coluna 'Days_from_reference'
unique_values = df['Days_from_reference'].unique()
print(unique_values)

# # Agrupar os dados por 'Days_from_reference', usando mediana para numéricas e moda para categóricas
# df = df.groupby('Days_from_reference').agg(
#     lambda x: x.mode()[0] if x.dtype == 'O' or x.dtype.name == 'category' else x.median()
# ).reset_index()

# Definir as features (X) e a variável target (y)
X = df.drop(columns=['Screening_date', 'Days_from_reference'])
y = df['Days_from_reference']

print(y.head(100))

# # Dividir os dados em conjuntos de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Definindo as datas de corte
# split_date = '2024-03-01'

# # Dividindo os dados com base na data
# train = df[df['Screening_date'] < split_date]
# test = df[df['Screening_date'] >= split_date]

# # Separando X e y para cada conjunto
# X_train = train.drop(columns=['Days_from_reference', 'Screening_date'])
# y_train = train['Days_from_reference']

# X_test = test.drop(columns=['Days_from_reference', 'Screening_date'])
# y_test = test['Days_from_reference']


####################### MLR ######################

print(X_test.columns)

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
# Modelo: MLR with random split without groupping in days
# MAE: 201.81401261891844
# MSE: 57377.137719109625
# RMSE: 239.53525360395204
# R2: 0.07137554928711676
# ME: -9.142030109721713
# MAV: 515.473496128648
# MPV: 0.8198384506200496
# RME: -0.5675761700262184
# RMAE: 0.8198384506200496

# Modelo: LR with random split with groupping in days
# MAE: 187.24869875244283
# MSE: 54520.05605471468
# RMSE: 233.49530199709517
# R2: -0.17124228819489207
# ME: -76.9537212792579
# MAV: 412.55555555555554
# MPV: 1.0827352295392714
# RME: -0.9279345266676897
# RMAE: 1.0827352295392714

# Modelo: LR with split date and groupping in days
# MAE: 402.49262830253065
# MSE: 167340.73282588026
# RMSE: 409.07301649690885
# R2: -509.185161054513
# ME: 402.49262830253065
# MAV: 820.0
# MPV: 0.4894598230979161
# RME: 0.4894598230979161
# RMAE: 0.4894598230979161

# Modelo: LR with groupping and no shuffle > no groupping
# MAE: 410.9561281858482
# MSE: 174534.71187328605
# RMSE: 417.77351743891813
# R2: -67.72305473119228
# ME: 410.9561281858482
# MAV: 769.6111111111111
# MPV: 0.5306799260231291
# RME: 0.5306799260231291
# RMAE: 0.5306799260231291

####################### LASSO ######################

from sklearn.linear_model import Lasso

# Criar o modelo Lasso (já com intercepto)
lasso_model = Lasso(alpha=1.0, fit_intercept=True)  # O parâmetro alpha controla a força da regularização

# Ajustar o modelo aos dados de treino
lasso_model.fit(X_train, y_train)

# Fazer previsões
y_pred = lasso_model.predict(X_test)

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

# print("Modelo: Lasso")
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

# Modelo: Lasso with split date and groupping
# MAE: 404.9036362695814
# MSE: 168763.87136436155
# RMSE: 410.80880146895777
# R2: -513.5239980620779
# ME: 404.9036362695814
# MAV: 820.0
# MPV: 0.49249666454012925
# RME: 0.49249666454012925
# RMAE: 0.49249666454012925

# Modelo: Lasso with split date and no groupping
# MAE: 359.3522122799821
# MSE: 133804.45677224518
# RMSE: 365.79291514769005
# R2: -440.74780468424456
# ME: 359.3522122799821
# MAV: 817.939393939394
# MPV: 0.43863504669186326
# RME: 0.43863504669186326
# RMAE: 0.43863504669186326

# Modelo: Lasso with groupping and random split
# MAE: 404.9036362695814
# MSE: 168763.87136436155
# RMSE: 410.80880146895777
# R2: -513.5239980620779
# ME: 404.9036362695814
# MAV: 820.0
# MPV: 0.49249666454012925
# RME: 0.49249666454012925
# RMAE: 0.49249666454012925

# Modelo: Lasso with random split and no groupping
# MAE: 359.3522122799821
# MSE: 133804.45677224518
# RMSE: 365.79291514769005
# R2: -440.74780468424456
# ME: 359.3522122799821
# MAV: 817.939393939394
# MPV: 0.43863504669186326
# RME: 0.43863504669186326
# RMAE: 0.43863504669186326

# Modelo: Lasso with groupping and no shuffle > no groupping
# MAE: 403.998654495718
# MSE: 168614.12628609105
# RMSE: 410.62650460739997
# R2: -65.39182375150664
# ME: 403.998654495718
# MAV: 769.6111111111111
# MPV: 0.5216048190361109
# RME: 0.5216048190361109
# RMAE: 0.5216048190361109

####################### RIDGE ######################

from sklearn.linear_model import Ridge

# Criar o modelo Ridge
ridge_model = Ridge(alpha=1.0)  # O parâmetro alpha controla a força da regularização

# Ajustar o modelo aos dados de treino
ridge_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = ridge_model.predict(X_test)

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

# print("Modelo: Ridge")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

# Modelo: Ridge with random split and groupping
# MAE: 186.2870690080856
# MSE: 53778.5150060029
# RMSE: 231.90195127683359
# R2: -0.15531192609453037
# ME: -76.20004301016596
# MAV: 412.55555555555554
# MPV: 1.0746852018190447
# RME: -0.9202532688745523
# RMAE: 1.0746852018190447

# Modelo: Ridge with split date and groupping
# MAE: 402.2484940602708
# MSE: 166646.7587699731
# RMSE: 408.22390764134957
# R2: -507.0693864938204
# ME: 402.2484940602708
# MAV: 820.0
# MPV: 0.4892383566409738
# RME: 0.4892383566409738
# RMAE: 0.4892383566409738

# Modelo: Ridge with random split and without groupping
# MAE: 201.79420980022272
# MSE: 57375.9022987365
# RMSE: 239.53267480395343
# R2: 0.0713955440378311
# ME: -9.112499259362561
# MAV: 515.473496128648
# MPV: 0.8198557949606539
# RME: -0.5675831192898458
# RMAE: 0.8198557949606539

# Modelo: Ridge with split date and without groupping
# MAE: 358.2938638907393
# MSE: 133955.10963797907
# RMSE: 365.9987836564202
# R2: -441.2451765529606
# ME: 358.2938638907393
# MAV: 817.939393939394
# MPV: 0.4372853815805343
# RME: 0.4372853815805343
# RMAE: 0.4372853815805343

# Modelo: Ridge with groupping and no shuffle > no groupping
# MAE: 403.85553055080067
# MSE: 168662.35125735062
# RMSE: 410.68522162034344
# R2: -65.41081233723672
# ME: 403.85553055080067
# MAV: 769.6111111111111
# MPV: 0.5213358933032334
# RME: 0.5213358933032334
# RMAE: 0.5213358933032334


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

# print("Modelo: GLM com Gaussian")
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
# Modelo: GLM com Gaussian with split date and groupping days
# MAE: 202.00974660777
# MSE: 57394.5596959847
# RMSE: 239.57161705006857
# R2: 0.07109358203763227
# ME: -8.624744936217366
# MAV: 515.473496128648
# MPV: 0.8191397791986705
# RME: -0.5660529714794953
# RMAE: 0.8191397791986705

# Modelo: GLM com Gaussian with split date without groupping by days
# MAE: 358.13556302361127
# MSE: 133954.6808604432
# RMSE: 365.9981978923437
# R2: -441.24376096831
# ME: 358.13556302361127
# MAV: 817.939393939394
# MPV: 0.4370924476188683
# RME: 0.4370924476188683
# RMAE: 0.4370924476188683

# Modelo: GLM com Gaussian with random split groupping by days
# MAE: 187.24869875244283
# MSE: 54520.05605471468
# RMSE: 233.49530199709517
# R2: -0.17124228819489207
# ME: -76.9537212792579
# MAV: 412.55555555555554
# MPV: 1.0827352295392714
# RME: -0.9279345266676897
# RMAE: 1.0827352295392714

# Modelo: GLM gaussian with groupping and no shuffle > no groupping
# MAE: 410.9561281858482
# MSE: 174534.71187328605
# RMSE: 417.77351743891813
# R2: -67.72305473119228
# ME: 410.9561281858482
# MAV: 769.6111111111111
# MPV: 0.5306799260231291
# RME: 0.5306799260231291
# RMAE: 0.5306799260231291


### Kernel Polinomial ###

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

# print(X_train_poly_sm[:5, :])
# print(X_test_poly_sm[:5, :])

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

# Modelo: GLM com Kernel Polinomial with random split groupping by days
# MAE: 222.78936216844195
# MSE: 71974.76617012703
# RMSE: 268.2811327136648
# R2: -0.5462179594384691
# ME: -70.26890692525528
# MAV: 412.55555555555554
# MPV: 1.1905396621015636
# RME: -0.9645662965997857
# RMAE: 1.1905396621015636

# Modelo: GLM com Kernel Polinomial with random split without groupping by days
# MAE: 193.28152727322066
# MSE: 54064.619424262964
# RMSE: 232.51799806523144
# R2: 0.12498724210259604
# ME: -8.722449399706955
# MAV: 515.473496128648
# MPV: 0.7862495306417968
# RME: -0.538244812224109
# RMAE: 0.7862495306417968

# Modelo: GLM com Kernel Polinomial with split date without groupping by days
# MAE: 337.35378821879874
# MSE: 122977.48998491054
# RMSE: 350.681465128841
# R2: -405.0031895565489
# ME: 337.18494519558186
# MAV: 817.939393939394
# MPV: 0.41162261160664865
# RME: 0.4114091564571961
# RMAE: 0.41162261160664865

# Modelo: GLM com Kernel Polinomial with split date with groupping by days
# MAE: 432.7254718933802
# MSE: 218963.2492453241
# RMSE: 467.93509084628835
# R2: -666.5708818455004
# ME: 432.7254718933802
# MAV: 820.0
# MPV: 0.5255523733269655
# RME: 0.5255523733269655
# RMAE: 0.5255523733269655

# Modelo: GLM kernel with groupping and no shuffle > no groupping
# MAE: 576.7701788700184
# MSE: 391193.12530377414
# RMSE: 625.454335106708
# R2: -153.03231983008325
# ME: 224.50363126126277
# MAV: 769.6111111111111
# MPV: 0.752891055611214
# RME: 0.26797552340159386
# RMAE: 0.752891055611214


####################### SVR KERNEL GAUSSIANO ######################

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Normalizar a coluna 'Age'
scaler = StandardScaler()
X['Age'] = scaler.fit_transform(X[['Age']])

# # Dividir os dados em conjuntos de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Definindo as datas de corte
# split_date = '2024-03-01'

# # Dividindo os dados com base na data
# train = df[df['Screening_date'] < split_date]
# test = df[df['Screening_date'] >= split_date]

# # Separando X e y para cada conjunto
# X_train = train.drop(columns=['Days_from_reference', 'Screening_date'])
# y_train = train['Days_from_reference']

# X_test = test.drop(columns=['Days_from_reference', 'Screening_date'])
# y_test = test['Days_from_reference']

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

# Modelo: SVR with random split with groupping by days
# MAE: 195.57475747524586
# MSE: 56067.85241694872
# RMSE: 236.7865123205896
# R2: -0.204493254612524
# ME: -95.78211637206364
# MAV: 412.55555555555554
# MPV: 1.1162160002317796
# RME: -0.9786985980951827
# RMAE: 1.1162160002317796

# Modelo: SVR with random split without groupping by days
# MAE: 201.2461009775297
# MSE: 58627.18636201044
# RMSE: 242.1305151401005
# R2: 0.05114404627870617
# ME: -36.43328625843166
# MAV: 515.473496128648
# MPV: 0.869375066261106
# RME: -0.6566453390665784
# RMAE: 0.869375066261106

# Modelo: SVR with split date without groupping by days
# MAE: 307.2210133240143
# MSE: 94695.70523248233
# RMSE: 307.7266729298621
# R2: -311.6324855582274
# ME: 307.2210133240143
# MAV: 817.939393939394
# MPV: 0.37532330904107397
# RME: 0.37532330904107397
# RMAE: 0.37532330904107397

# Modelo: SVR with split date with groupping days
# MAE: 365.1020912873053
# MSE: 133627.73119067523
# RMSE: 365.5512702627023
# R2: -406.40161948376596
# ME: 365.1020912873053
# MAV: 820.0
# MPV: 0.44497538902093403
# RME: 0.44497538902093403
# RMAE: 0.44497538902093403

# Modelo: SVR with groupping and no shuffle > no groupping
# MAE: 346.64240201620214
# MSE: 122724.0278361908
# RMSE: 350.3198935775569
# R2: -47.32259434983942
# ME: 346.64240201620214
# MAV: 769.6111111111111
# MPV: 0.4479676933998404
# RME: 0.4479676933998404
# RMAE: 0.4479676933998404


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

print("Modelo: SVR")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

### results

# Modelo: SVR no shuffle and groupping
# MAE: 346.9327148384151
# MSE: 123134.01472796743
# RMSE: 350.9045664108226
# R2: -47.484026716502925
# ME: 346.9327148384151
# MAV: 769.6111111111111
# MPV: 0.448178066557626
# RME: 0.448178066557626
# RMAE: 0.448178066557626

# Modelo: SVR split date and groupping
# MAE: 373.8616724738721
# MSE: 140202.46029319614
# RMSE: 374.43618988179566
# R2: -426.44652528413457
# ME: 373.8616724738721
# MAV: 820.0
# MPV: 0.45559453124149024
# RME: 0.45559453124149024
# RMAE: 0.45559453124149024

# Modelo: SVR split date and no groupping
# MAE: 326.3997381697746
# MSE: 115598.82054527928
# RMSE: 339.9982655033394
# R2: -380.64293202046395
# ME: 326.3997381697746
# MAV: 817.939393939394
# MPV: 0.3981100006008843
# RME: 0.3981100006008843
# RMAE: 0.3981100006008843

# Modelo: SVR no shuffle and no groupping
# MAE: 332.97058542613206
# MSE: 119878.27822419783
# RMSE: 346.23442668833184
# R2: -344.22499723273995
# ME: 332.97058542613206
# MAV: 816.0190589636688
# MPV: 0.4073427754335244
# RME: 0.4073427754335244
# RMAE: 0.4073427754335244