# Regression_Screenings_Count SEM ZEROS!!!!!!!!!!!!!!!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Carregar o DataFrame
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

# print(df)

# Adicionar coluna com o número total de screenings por dia
df['Total_screenings'] = df.groupby('Screening_date')['Screening_date'].transform('count')

lenght_data = len('Total_screenings')
print({lenght_data})

# Agrupar os dados por dias, usando mediana para numéricas e moda para categóricas
aggregations = {col: (lambda x: x.mode()[0] if x.dtype == 'O' else x.median()) for col in df.columns if col not in ['Screening_date', 'Total_screenings']}
aggregations['Total_screenings'] = 'first'

df = df.groupby('Screening_date').agg(aggregations).reset_index()

print(df)


### STATIONARITY ###

# # Plotar a série temporal dos screenings diários
# plt.figure(figsize=(14, 7))
# plt.plot(df['Screening_date'], df['Total_screenings'], label='Total Screenings', color='blue')
# plt.title('Screenings Diários')
# plt.xlabel('Data')
# plt.ylabel('Total de Screenings')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# from statsmodels.tsa.stattools import adfuller

# # Extraia os valores da série temporal para o teste
# screenings_series = df.set_index('Screening_date')['Total_screenings']

# print(screenings_series.head())

# # Realizar o teste ADF
# result = adfuller(screenings_series.dropna(), autolag='AIC')  # Usar autolag para escolher o número de defasagens

# # Exibir resultados do teste
# print('Estatística do Teste ADF:', result[0])
# print('P-Valor:', result[1])
# print('Número de defasagens usadas no teste:', result[2])
# print('Número de observações usadas no teste:', result[3])
# print('Valores críticos:')
# for key, value in result[4].items():
#     print(f'\t{key}: {value}')

# # aplicando diferenciação
# df['Differenced_1'] = df['Total_screenings'].diff().dropna()
# df['Differenced_2'] = df['Differenced_1'].diff().dropna()

# # Realizar o teste ADF após a segunda diferenciação
# result_2 = adfuller(df['Differenced_2'].dropna(), autolag='AIC')

# # Exibir resultados do teste
# print('Estatística do Teste ADF (2ª Diferença):', result_2[0])
# print('P-Valor:', result_2[1])
# print('Número de defasagens usadas no teste:', result_2[2])
# print('Número de observações usadas no teste:', result_2[3])
# print('Valores críticos:')
# for key, value in result_2[4].items():
#     print(f'\t{key}: {value}')

# Estatística do Teste ADF (2ª Diferença): -8.427356727606558
# P-Valor: 1.910865522717346e-13
# Número de defasagens usadas no teste: 7
# Número de observações usadas no teste: 80
# Valores críticos:
#         1%: -3.5148692050781247
#         5%: -2.8984085156250003
#         10%: -2.58643890625

# # Remover os valores nulos da série diferenciada
# df = df.dropna(subset=['Differenced_2'])

# # Substituir a coluna 'Total_screenings' pela série diferenciada
# df['Total_screenings'] = df['Differenced_2']

# # Remover a coluna auxiliar 'Differenced_1' e 'Differenced_2' (opcional)
# df = df.drop(columns=['Differenced_1', 'Differenced_2'])

# # Exibir o DataFrame atualizado
# print(df.columns)

# Definir as features (X) e a variável target (y)
X = df.drop(columns=['Screening_date', 'Total_screenings'])
y = df['Total_screenings']

print(y.head(100))

# # Dividir os dados em conjuntos de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Dividir os dados em conjuntos de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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

####################### MLR ######################

from sklearn.linear_model import LinearRegression

# # Adicionar uma constante aos dados
# X_train_sm = sm.add_constant(X_train)
# X_test_sm = X_test.copy()  # Certifique-se de que não está sobrescrevendo X_test original
# X_test_sm.insert(0, 'const', 1.0)

# print(X_train_sm.head())
# print(X_test_sm.head())

# # Ajustar o modelo
# model_sm = sm.OLS(y_train, X_train_sm).fit()

# # # Obter o resumo do modelo
# # print(model_sm.summary())

# # Fazer previsões no conjunto de teste
# y_pred = model_sm.predict(X_test_sm)

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

# Modelo: LR with split date with groupping by days
# MAE: 124.70943152559461
# MSE: 20535.944923140174
# RMSE: 143.30368077317544
# R2: -2.4493134057542245
# ME: 124.70943152559461
# MAV: 226.28571428571428
# MPV: 0.5241509500056887
# RME: 0.5241509500056887
# RMAE: 0.5241509500056887

# Modelo: LR with random split with groupping by days
# MAE: 35.23940029445204
# MSE: 2262.9570344969375
# RMSE: 47.57054797347764
# R2: 0.034035351372527556
# ME: -8.317020855954707
# MAV: 78.55555555555556
# MPV: 1.7647136876328624
# RME: -1.5877663752136364
# RMAE: 1.7647136876328624

# Modelo: LR without groupping by days with random split
# MAE: 61.45809318181652
# MSE: 6704.8693310512435
# RMSE: 81.88326648987108
# R2: 0.10777760079001464
# ME: 0.7685869404160891
# MAV: 142.05717689100655
# MPV: 0.6829943848141611
# RME: -0.4469226966187693
# RMAE: 0.6829943848141611

# Modelo: LR with split date without groupping days
# MAE: 132.62228149155132
# MSE: 25911.203261348055
# RMSE: 160.96957247053885
# R2: -1.9413527262261172
# ME: 132.62228149155132
# MAV: 252.59595959595958
# MPV: 0.4771039183568591
# RME: 0.4771039183568591
# RMAE: 0.4771039183568591

# Modelo: LR with groupping and no shuffle > no groupping
# MAE: 98.40667174701024
# MSE: 14731.724363298637
# RMSE: 121.37431508889613
# R2: -0.7090552992583321
# ME: 87.3617099059246
# MAV: 167.83333333333334
# MPV: 1.026153212357035
# RME: -0.16052099426751337
# RMAE: 1.026153212357035


####################### GLM ######################

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Gaussian
from sklearn.preprocessing import PolynomialFeatures

### Gaussian ###

# # Ajustar o modelo GLM com família Gaussian
# model_glm = GLM(y_train, X_train_sm, family=Gaussian()).fit()

# # # Obter o resumo do modelo
# # print(model_glm.summary())

# # Fazer previsões no conjunto de teste
# y_pred = model_glm.predict(X_test_sm)

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

# Modelo: GLM com Gaussian with random split with groupping by days
# MAE: 35.23940029445204
# MSE: 2262.9570344969375
# RMSE: 47.57054797347764
# R2: 0.034035351372527556
# ME: -8.317020855954707
# MAV: 78.55555555555556
# MPV: 1.7647136876328624
# RME: -1.5877663752136364
# RMAE: 1.7647136876328624

# Modelo: GLM com Gaussian with split date with groupping by days
# MAE: 124.70943152559461
# MSE: 20535.944923140174
# RMSE: 143.30368077317544
# R2: -2.4493134057542245
# ME: 124.70943152559461
# MAV: 226.28571428571428
# MPV: 0.5241509500056887
# RME: 0.5241509500056887
# RMAE: 0.5241509500056887

# Modelo: GLM com Gaussian without intercept
# MAE: 34.62795932828225
# MSE: 2227.9325929519546
# RMSE: 47.200980847350564
# R2: 0.04898586605514221
# ME: -4.675987881360612
# MAV: 78.55555555555556
# MPV: 1.5720236186216325
# RME: -1.364870991214165
# RMAE: 1.5720236186216325

# Modelo: GLM com Gaussian without groupping by days with random split
# MAE: 61.45809318181652
# MSE: 6704.8693310512435
# RMSE: 81.88326648987108
# R2: 0.10777760079001464
# ME: 0.7685869404160891
# MAV: 142.05717689100655
# MPV: 0.6829943848141611
# RME: -0.4469226966187693
# RMAE: 0.6829943848141611

# Modelo: GLM com Gaussian with split date without groupping days
# MAE: 132.62228149155132
# MSE: 25911.203261348055
# RMSE: 160.96957247053885
# R2: -1.9413527262261172
# ME: 132.62228149155132
# MAV: 252.59595959595958
# MPV: 0.4771039183568591
# RME: 0.4771039183568591
# RMAE: 0.4771039183568591

# Modelo: GLM Gaussian with groupping and no shuffle > no groupping
# MAE: 98.40667174701024
# MSE: 14731.724363298637
# RMSE: 121.37431508889613
# R2: -0.7090552992583321
# ME: 87.3617099059246
# MAV: 167.83333333333334
# MPV: 1.026153212357035
# RME: -0.16052099426751337
# RMAE: 1.026153212357035

### Kernel Polinomial ###

# Definir o grau do polinômio
# degree = 2

# # Criar o transformador polinomial
# poly = PolynomialFeatures(degree)

# # Ajustar e transformar os dados de treino
# X_train_poly = poly.fit_transform(X_train)
# X_test_poly = poly.transform(X_test)

# # Adicionar uma constante aos dados transformados
# X_train_poly_sm = sm.add_constant(X_train_poly)
# X_test_poly_sm = sm.add_constant(X_test_poly)

# # Ajustar o modelo GLM com família Gaussian
# model_glm = GLM(y_train, X_train_poly_sm, family=Gaussian()).fit()

# # # Obter o resumo do modelo
# # print(model_glm.summary())

# # Fazer previsões no conjunto de teste
# y_pred = model_glm.predict(X_test_poly_sm)

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

# Modelo: GLM com Kernel Polinomial with split date with groupping by days
# MAE: 127.91475051437033
# MSE: 22292.39110181504
# RMSE: 149.3063665816533
# R2: -2.7443343250868506
# ME: 127.91475051437033
# MAV: 226.28571428571428
# MPV: 0.5333508840835379
# RME: 0.5333508840835379
# RMAE: 0.5333508840835379

# Modelo: GLM com Kernel Polinomial with random split with groupping by days
# MAE: 47.213114661232446
# MSE: 3774.840476130728
# RMSE: 61.43973043667044
# R2: -0.6113264187364378
# ME: -6.549899592652912
# MAV: 78.55555555555556
# MPV: 1.9555278607966393
# RME: -0.672844418567795
# RMAE: 1.9555278607966393

# Modelo: GLM com Kernel Polinomial without groupping by days with random split
# MAE: 58.68691208515383
# MSE: 6445.467508032106
# RMSE: 80.28366899956744
# R2: 0.1422964117415908
# ME: 0.9001594259145399
# MAV: 142.05717689100655
# MPV: 0.6392851861581772
# RME: -0.41029217336308477
# RMAE: 0.6392851861581772

# Modelo: GLM com Kernel Polinomial with split date without groupping days
# MAE: 127.57460340439565
# MSE: 25195.455532581807
# RMSE: 158.73076429155694
# R2: -1.8601034491446011
# ME: 127.57460340439565
# MAV: 252.59595959595958
# MPV: 0.45128601388089923
# RME: 0.45128601388089923
# RMAE: 0.45128601388089923

# Modelo: GLM kernel with groupping and no shuffle > no groupping
# MAE: 108.90981506311219
# MSE: 19156.68355954431
# RMSE: 138.40767160654178
# R2: -1.2224032127032873
# ME: 55.90757822125236
# MAV: 167.83333333333334
# MPV: 1.492704634337572
# RME: -0.7602562476502198
# RMAE: 1.492704634337572


####################### SVR KERNEL GAUSSIANO ######################

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# # Normalizar a coluna 'Age'
# scaler = StandardScaler()
# X['Age'] = scaler.fit_transform(X[['Age']])

# # Dividir os dados em conjuntos de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Dividir os dados em conjuntos de treinamento e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

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

# # Definir o modelo SVR com kernel RBF
# model_svr = SVR(kernel='rbf', C=1.0, gamma='scale')

# # Ajustar o modelo
# model_svr.fit(X_train, y_train)

# "Summary" of the model
# print(f"Support Vectors: {model_svr.support_vectors_}")
# print(f"Number of Support Vectors: {model_svr.n_support_}")
# print(f"Dual Coefficients: {model_svr.dual_coef_}")

# # Fazer previsões no conjunto de teste
# y_pred = model_svr.predict(X_test)

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

# Modelo: SVR without groupping by days with random split
# MAE: 58.44080004565813
# MSE: 8040.683714264781
# RMSE: 89.66985956420798
# R2: -0.06998030246551723
# ME: 33.589573661387796
# MAV: 142.05717689100655
# MPV: 0.5229535823443424
# RME: -0.14249781260817707
# RMAE: 0.5229535823443424

# Modelo: SVR with random splite with groupping days
# MAE: 34.54291114131281
# MSE: 2413.5464125637727
# RMSE: 49.12785780556458
# R2: -0.03024515128566696
# ME: -10.175000031590061
# MAV: 78.55555555555556
# MPV: 4.0835652156942155
# RME: -3.9228700822933473
# RMAE: 4.0835652156942155

# Modelo: SVR with splite date with groupping days
# MAE: 139.45066680403
# MSE: 25398.03201261867
# RMSE: 159.36760025996085
# R2: -3.2659723050866374
# ME: 139.45066680403
# MAV: 226.28571428571428
# MPV: 0.5850977552691745
# RME: 0.5850977552691745
# RMAE: 0.5850977552691745

# Modelo: SVR with splite date without groupping days
# MAE: 157.16554242160228
# MSE: 33488.52179250764
# RMSE: 182.99869341748766
# R2: -2.8015044642334552
# ME: 157.16554242160228
# MAV: 252.59595959595958
# MPV: 0.5784121065361402
# RME: 0.5784121065361402
# RMAE: 0.5784121065361402

# Modelo: SVR kernel with groupping and no shuffle > no groupping
# MAE: 99.88408753233892
# MSE: 15324.566684639385
# RMSE: 123.79243387477034
# R2: -0.7778320619729684
# ME: 82.5486032043014
# MAV: 167.83333333333334
# MPV: 1.256449226745897
# RME: -0.43973318443726855
# RMAE: 1.256449226745897


####################### SVR LINEAR KERNEL ######################

# # Definir o modelo SVR com kernel RBF
# model_svr_linear = SVR(kernel='linear', C=1.0)

# # Ajustar o modelo
# model_svr_linear.fit(X_train, y_train)

# "Summary" of the model
# print(f"Support Vectors: {model_svr_linear.support_vectors_}")
# print(f"Number of Support Vectors: {model_svr_linear.n_support_}")
# print(f"Dual Coefficients: {model_svr_linear.dual_coef_}")

# # Fazer previsões no conjunto de teste
# y_pred = model_svr_linear.predict(X_test)

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
# Modelo: SVR with groupping and split date
# MAE: 137.9811299531727
# MSE: 24829.99420754794
# RMSE: 157.5753604074823
# R2: -3.170562017255282
# ME: 137.9811299531727
# MAV: 226.28571428571428
# MPV: 0.5787681475586259
# RME: 0.5787681475586259
# RMAE: 0.5787681475586259

# Modelo: SVR with groupping and no shuffle
# MAE: 99.32602627000936
# MSE: 15233.271291803198
# RMSE: 123.42313920737553
# R2: -0.7672407102020056
# ME: 82.44747462000024
# MAV: 167.83333333333334
# MPV: 1.2483621688063897
# RME: -0.43589414806427235
# RMAE: 1.2483621688063897

# Modelo: SVR no groupping and no shuffle
# MAE: 152.50885101467597
# MSE: 31412.317028755577
# RMSE: 177.2352025664077
# R2: -2.7112080555306757
# ME: 152.50885101467597
# MAV: 249.56343061346038
# MPV: 0.5705098857326845
# RME: 0.5705098857326845
# RMAE: 0.5705098857326845

# Modelo: SVR no groupping and split date
# MAE: 154.64650082901196
# MSE: 32467.102344139224
# RMSE: 180.18629899118085
# R2: -2.6855563606746964
# ME: 154.64650082901196
# MAV: 252.59595959595958
# MPV: 0.5695032114965531
# RME: 0.5695032114965531
# RMAE: 0.5695032114965531


####################### MLP NN: Multilayer Perceptron Neural Network ######################

from sklearn.neural_network import MLPRegressor

# # Definir o modelo MLP com uma camada oculta de 100 neurônios (você pode ajustar os parâmetros conforme necessário)
# model_mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# # Ajustar o modelo aos dados de treino
# model_mlp.fit(X_train, y_train)

# # "Summary" do modelo
# print(f"Coefs: {model_mlp.coefs_}")
# print(f"Intercepts: {model_mlp.intercepts_}")
# print(f"Número de iterações: {model_mlp.n_iter_}")

# # Fazer previsões no conjunto de teste
# y_pred = model_mlp.predict(X_test)

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

# Modelo: MPL NN groupping and no shuffle
# MAE: 100.43273272928516
# MSE: 15137.957445059288
# RMSE: 123.03640699020468
# R2: -0.7561831699675305
# ME: 89.10933066452216
# MAV: 167.83333333333334
# MPV: 1.0412005738952947
# RME: -0.15160094446637143
# RMAE: 1.0412005738952947

# Modelo: MPL NN groupping and split date
# MAE: 132.14320302611247
# MSE: 22852.74232197113
# RMSE: 151.17123510103082
# R2: -2.838453538147127
# ME: 132.14320302611247
# MAV: 226.28571428571428
# MPV: 0.5537965371908629
# RME: 0.5537965371908629
# RMAE: 0.5537965371908629

# Modelo: MPL NN no groupping and split date
# MAE: 128.75434422734685
# MSE: 25217.501653446685
# RMSE: 158.8001941228243
# R2: -1.8626060507048048
# ME: 128.75434422734685
# MAV: 252.59595959595958
# MPV: 0.457965385292423
# RME: 0.457965385292423
# RMAE: 0.457965385292423

# Modelo: MPL NN no groupping and no shuffle
# MAE: 129.69612766465224
# MSE: 24856.625557170093
# RMSE: 157.65984129501746
# R2: -1.9366859158028116
# ME: 129.69612766465224
# MAV: 249.56343061346038
# MPV: 0.47440337656748094
# RME: 0.47440337656748094
# RMAE: 0.47440337656748094


####################### LSTM ######################

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Certifique-se de que seus dados estejam no formato adequado para o LSTM
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

# LSTM groupping and no shuffle
# MAE: 146.11372746361627
# MSE: 29398.674707228776
# RMSE: 171.4604173190675
# R2: -2.4105958938598633
# ME: 144.14877902136908
# MAV: 167.83333333333334
# MPV: 0.9242718478067466
# RME: 0.5967804407655501
# RMAE: 0.9242718478067466

# LSTM groupping and split date
# MAE: 202.28013719831193
# MSE: 46870.88644031851
# RMSE: 216.4968508785255
# R2: -6.872654438018799
# ME: 202.28013719831193
# MAV: 226.28571428571428
# MPV: 0.8852954215556175
# RME: 0.8852954215556175
# RMAE: 0.8852954215556175

# LSTM no groupping and split date
# MAE: 131.4321320899809
# MSE: 26139.33430400483
# RMSE: 161.6766349971598
# R2: -1.9672493934631348
# ME: 131.4321320899809
# MAV: 252.59595959595958
# MPV: 0.467300733218471
# RME: 0.467300733218471
# RMAE: 0.467300733218471

# LSTM no groupping and no shuffle
# MAE: 129.25905747723195
# MSE: 25128.808550050788
# RMSE: 158.52068808218942
# R2: -1.9688427448272705
# ME: 129.25905747723195
# MAV: 249.56343061346038
# MPV: 0.4689218598885342
# RME: 0.4689218598885342
# RMAE: 0.4689218598885342


####################### XGBOOST ######################

from xgboost import XGBRegressor

# # Definir o modelo XGBoost
# model_xgb = XGBRegressor(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )

# # Ajustar o modelo aos dados de treino
# model_xgb.fit(X_train, y_train)

# # Fazer previsões no conjunto de teste
# y_pred = model_xgb.predict(X_test)

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

# Modelo: XGB groupping and no shuffle
# MAE: 99.30447726779514
# MSE: 14842.117162216578
# RMSE: 121.82822810094784
# R2: -0.7218620777130127
# ME: 80.27742513020833
# MAV: 167.83333333333334
# MPV: 1.1831435424614747
# RME: -0.3559534708237185
# RMAE: 1.1831435424614747

# Modelo: XGB groupping and split date
# MAE: 122.99813079833984
# MSE: 21585.19282364344
# RMSE: 146.91900089383756
# R2: -2.6255505084991455
# ME: 122.99813079833984
# MAV: 226.28571428571428
# MPV: 0.5100461229531309
# RME: 0.5100461229531309
# RMAE: 0.5100461229531309

# Modelo: XGB no groupping and split date
# MAE: 130.2766845828355
# MSE: 26468.65217920067
# RMSE: 162.6918934034535
# R2: -2.0046327114105225
# ME: 129.74500173029273
# MAV: 252.59595959595958
# MPV: 0.46158065415391836
# RME: 0.4588139183530801
# RMAE: 0.46158065415391836

# Modelo: XGB no groupping and no shuffle
# MAE: 130.2705766906193
# MSE: 25932.60829937145
# RMSE: 161.03604658389827
# R2: -2.063807725906372
# ME: 129.71565053915964
# MAV: 249.56343061346038
# MPV: 0.47242770600313044
# RME: 0.4695560358279342
# RMAE: 0.47242770600313044


####################### CNN ######################

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# # Definindo as datas de corte
# split_date = '2024-01-01'

# # Dividindo os dados com base na data
# train = df[df['Screening_date'] < split_date]
# test = df[df['Screening_date'] >= split_date]

# # Separando X e y para cada conjunto
# X_train = train.drop(columns=['Total_screenings', 'Screening_date'])
# y_train = train['Total_screenings']

# X_test = test.drop(columns=['Total_screenings', 'Screening_date'])
# y_test = test['Total_screenings']

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

print("Modelo: CNN")
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

# Modelo: CNN groupping and no shuffle
# MAE: 127.65773362379808
# MSE: 22865.09487653428
# RMSE: 151.21208574890528
# R2: -2.4225542545318604
# ME: 127.20328463040866
# MAV: 199.46153846153845
# MPV: 0.5643577409870968
# RME: 0.5547236825456265
# RMAE: 0.5643577409870968

# Modelo: CNN groupping and split date
# MAE: 136.90972137451172
# MSE: 25593.34213058135
# RMSE: 159.97919280513122
# R2: -2.7770845890045166
# ME: 136.90972137451172
# MAV: 206.77777777777777
# MPV: 0.6018035635433666
# RME: 0.6018035635433666
# RMAE: 0.6018035635433666

# Modelo: CNN no groupping and split date
# MAE: 140.57225036621094
# MSE: 26395.423816462844
# RMSE: 162.4666852510472
# R2: -2.593053102493286
# ME: 139.68582860020072
# MAV: 230.25422509888529
# MPV: 0.5708021055607458
# RME: 0.5449078678654898
# RMAE: 0.5708021055607458

# Modelo: CNN no groupping and no shuffle
# MAE: 119.70601725834672
# MSE: 23276.098702119114
# RMSE: 152.5650638321864
# R2: -1.74424147605896
# ME: 118.58960771959457
# MAV: 249.71445639187576
# MPV: 0.423388470248212
# RME: 0.4162624635093238
# RMAE: 0.423388470248212
