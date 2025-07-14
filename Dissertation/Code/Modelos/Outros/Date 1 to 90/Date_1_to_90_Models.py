###################### REGRESSION 1 to 90 ##########################


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

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

# ################ groupping by days
# # Agrupar os dados por dias, usando mediana para numéricas e moda para categóricas
# df = df.groupby('Screening_date').agg(lambda x: x.mode()[0] if x.dtype == 'O' else x.median()).reset_index()

# print(df)

# # Número total de datas diferentes
# num_dates = df['Screening_date'].nunique()
# print(f'Total de datas diferentes: {num_dates}')

# # Criar uma nova coluna numérica contínua baseada nas datas
# df['Date'] = range(1, num_dates + 1)

# # Reordenar as colunas para colocar a nova coluna 'Date' em primeiro lugar
# cols = ['Date'] + [col for col in df.columns if col != 'Date']
# df = df[cols]

print(df)
print(df.columns)

############## without groupping by days
# Número total de datas diferentes
num_dates = df['Screening_date'].nunique()
print(f'Total de datas diferentes: {num_dates}')

# Criar um mapeamento de cada data única para um número contínuo
date_mapping = {date: i+1 for i, date in enumerate(sorted(df['Screening_date'].unique()))}

# Criar uma nova coluna 'Date' mapeando 'Screening_date' para números contínuos
df['Date'] = df['Screening_date'].map(date_mapping)

# Reordenar as colunas para colocar a nova coluna 'Date' em primeiro lugar
cols = ['Date'] + [col for col in df.columns if col != 'Date']
df = df[cols]

# Definir as features (X) e a variável target (y)
X = df.drop(columns=['Screening_date', 'Date'])
y = df['Date']

print(y)

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
# X_train = train.drop(columns=['Date', 'Screening_date'])
# y_train = train['Date']

# X_test = test.drop(columns=['Date', 'Screening_date'])
# y_test = test['Date']


####################### MLR ######################

from sklearn.linear_model import LinearRegression

# Adicionar uma constante aos dados
X_train_sm = sm.add_constant(X_train)

X_test_sm = X_test.copy()  # Certifique-se de que não está sobrescrevendo X_test original
X_test_sm.insert(0, 'const', 1.0)

# print(X_train_sm.head())
# print(X_test_sm.head())

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

# print("Modelo: MLR")
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
# Modelo: MLR with split date groupping by days
# MAE: 46.20807365679669
# MSE: 2206.1213144623825
# RMSE: 46.96936570215083
# R2: -550.5303286155956
# ME: 46.20807365679669
# MAV: 87.0
# MPV: 0.5296213972558631
# RME: 0.5296213972558631
# RMAE: 0.5296213972558631

# Modelo: MLR with random split without groupping by days
# MAE: 23.8820505422682
# MSE: 785.0418621538502
# RMSE: 28.01859850445504
# R2: 0.07831064452206338
# ME: -1.112767972780662
# MAV: 51.70756402620607
# MPV: 2.1595550289028336
# RME: -1.8754513779618545
# RMAE: 2.1595550289028336

# Modelo: MLR with split date without groupping by days
# MAE: 41.203940296629284
# MSE: 1775.5916936741328
# RMSE: 42.1377703927739
# R2: -458.8024447091781
# ME: 41.203940296629284
# MAV: 86.73989898989899
# MPV: 0.47423796432518
# RME: 0.47423796432518
# RMAE: 0.47423796432518

# Modelo: MLR groupping no shuffle = no groupping no shuffle
# MAE: 47.969160221122785
# MSE: 2357.430793012709
# RMSE: 48.55338086078773
# R2: -86.58256816146287
# ME: 47.969160221122785
# MAV: 81.5
# MPV: 0.585859854899805
# RME: 0.585859854899805
# RMAE: 0.585859854899805


# # Visualizar as previsões vs os valores reais
# plt.figure(figsize=(10, 6))

# # Criar um gráfico de dispersão
# plt.scatter(y_test.values, y_pred, color='blue', label='Previsões', alpha=0.6)

# # Adicionar linha de referência
# max_value = max(y_test.values.max(), y_pred.max())
# plt.plot([0, max_value], [0, max_value], color='red', linestyle='--', label='Valor Real (45°)')

# # Adicionar título e rótulos
# plt.title('Previsões vs Valores Reais')
# plt.xlabel('Valores Reais')
# plt.ylabel('Previsões')
# plt.xlim(0, max_value)
# plt.ylim(0, max_value)

# # Adicionar grade
# plt.grid(True)

# # Adicionar legenda
# plt.legend()

# # Mostrar o gráfico
# plt.show()

# # Gráfico de Importância das Variáveis
# importance = model.coef_  # Coeficientes do modelo
# feature_names = X.columns  # Nomes das variáveis

# # Criar um DataFrame para plotar
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importance
# }).sort_values(by='Importance', ascending=False)

# # Plotar
# # plt.figure(figsize=(10, 6))
# # plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
# # plt.xlabel('Importância')
# # plt.title('Importância das Variáveis')
# # plt.show()

# # Gráfico Residuals vs Fitted
# residuals = y_train - model_sm.fittedvalues
# fitted = model_sm.fittedvalues

# # Ajuste de uma linha de tendência nos resíduos
# slope, intercept = np.polyfit(fitted, residuals, 1)
# line = slope * fitted + intercept

# # Criação do gráfico
# plt.figure(figsize=(10, 6))
# plt.scatter(fitted, residuals, alpha=0.5, label='Resíduos')
# plt.axhline(y=0, color='red', linestyle='--', label='Linha de referência (y=0)')
# plt.plot(fitted, line, color='blue', linestyle='-', label='Linha de tendência')
# plt.title('Resíduos vs Valores Ajustados')
# plt.xlabel('Valores Ajustados')
# plt.ylabel('Resíduos')
# plt.grid()
# plt.legend()
# plt.show()

# # Q-Q Plot
# sm.qqplot(residuals, line='s')
# plt.title('Q-Q Plot dos Resíduos')
# plt.show()

# # Gráfico Resíduos vs Leverage
# leverage = model_sm.get_influence().hat_matrix_diag
# plt.figure(figsize=(10, 6))
# plt.scatter(leverage, residuals, alpha=0.5)
# plt.axhline(y=0, color='red', linestyle='--')
# plt.title('Resíduos vs Leverage')
# plt.xlabel('Leverage')
# plt.ylabel('Resíduos')
# plt.grid()
# plt.show()

# Função para BACKWARD ELIMINATION
def backward_elimination(X, y, significance_level=0.05):
    X_with_intercept = sm.add_constant(X)  # Adiciona a constante
    model = sm.OLS(y, X_with_intercept).fit()  # Ajusta o modelo
    while True:
        p_values = model.pvalues.iloc[1:]  # Ignora o p-value da constante
        max_p_value = p_values.max()  # Encontra o maior p-value
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()  # Identifica a variável a ser excluída
            X = X.drop(columns=excluded_feature)  # Remove a variável do modelo
            model = sm.OLS(y, sm.add_constant(X)).fit()  # Ajusta o modelo novamente
        else:
            break
    return model, X

# Executar backward elimination
model_sm, selected_features = backward_elimination(X_train, y_train)

# # Exibir o resumo do modelo final
# print(model_sm.summary())

# Fazer previsões no conjunto de teste
X_test_sm = sm.add_constant(X_test[selected_features.columns])  # Adiciona a constante ao conjunto de teste
y_pred = model_sm.predict(X_test_sm)

# print(X_test_sm)

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

# print("Modelo: MLR com backward elimination")
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
# Modelo: MLR com backward elimination with random split groupping by days
# MAE: 22.086577893568773
# MSE: 741.0805649822191
# RMSE: 27.22279495169846
# R2: -0.15730771259025977
# ME: -7.261808269407041
# MAV: 39.388888888888886
# MPV: 3.688779908851088
# RME: -3.489789549237064
# RMAE: 3.688779908851088

# Modelo: MLR com backward elimination with split date groupping by days
# MAE: 48.31294471811714
# MSE: 2370.9136338542553
# RMSE: 48.69202844259269
# R2: -591.7284084635638
# ME: 48.31294471811714
# MAV: 87.0
# MPV: 0.5545537763055373
# RME: 0.5545537763055373
# RMAE: 0.5545537763055373

# Modelo: MLR com backward elimination with random split without groupping by days
# MAE: 23.892573551270345
# MSE: 785.4474108729299
# RMSE: 28.02583470430328
# R2: 0.07783450438798445
# ME: -1.0622016749183638
# MAV: 51.70756402620607
# MPV: 2.1587697362922027
# RME: -1.8740993095216747
# RMAE: 2.1587697362922027

# Modelo: MLR com backward elimination with split date without groupping by days
# MAE: 41.15081159744375
# MSE: 1768.6389494628445
# RMSE: 42.05518932858161
# R2: -457.00198078654455
# ME: 41.15081159744375
# MAV: 86.73989898989899
# MPV: 0.4736251380701536
# RME: 0.4736251380701536
# RMAE: 0.4736251380701536

# Modelo: MLR com backward elimination no groupping no shuffle = groupping no shuffle
# MAE: 45.000000000000014
# MSE: 2051.916666666668
# RMSE: 45.29808678814888
# R2: -75.2321981424149
# ME: 45.000000000000014
# MAV: 81.5
# MPV: 0.5503190889614169
# RME: 0.5503190889614169
# RMAE: 0.5503190889614169


# Função para FORWARD SELECTION
def forward_selection(X, y, significance_level=0.05):
    selected_features = []  # Lista de variáveis selecionadas
    remaining_features = list(X.columns)  # Variáveis restantes
    while remaining_features:
        best_feature = None
        best_p_value = float('inf')
       
        for feature in remaining_features:
            # Ajustar o modelo com a nova variável
            model = sm.OLS(y, sm.add_constant(X[selected_features + [feature]])).fit()
            p_value = model.pvalues[feature]  # Obter o p-value da variável
           
            # Verificar se o p-value é o menor encontrado
            if p_value < best_p_value:
                best_p_value = p_value
                best_feature = feature
               
        if best_p_value < significance_level:
            selected_features.append(best_feature)  # Adiciona a variável ao modelo
            remaining_features.remove(best_feature)  # Remove a variável da lista
        else:
            break
           
    return selected_features

# Executar forward selection
selected_features = forward_selection(X_train, y_train)

# Ajustar o modelo final com as variáveis selecionadas
X_train_sm = sm.add_constant(X_train[selected_features])
model_sm = sm.OLS(y_train, X_train_sm).fit()

# # Exibir o resumo do modelo final
# print(model_sm.summary())

# Fazer previsões no conjunto de teste
X_test_sm = sm.add_constant(X_test[selected_features])  # Adiciona a constante ao conjunto de teste
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

# print("Modelo: MLR com forward selection")
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

# Modelo: MLR com forward selection with random split groupping by days
# MAE: 22.08657789356877
# MSE: 741.0805649822189
# RMSE: 27.222794951698454
# R2: -0.15730771259025933
# ME: -7.261808269407033
# MAV: 39.388888888888886
# MPV: 3.6887799088510866
# RME: -3.4897895492370634
# RMAE: 3.6887799088510866

# Modelo: MLR com forward selection with split date groupping by days
# MAE: 48.3129447181171
# MSE: 2370.913633854253
# RMSE: 48.692028442592665
# R2: -591.7284084635633
# ME: 48.3129447181171
# MAV: 87.0
# MPV: 0.5545537763055371
# RME: 0.5545537763055371
# RMAE: 0.5545537763055371

# Modelo: MLR com forward selection with random split without groupping by days
# MAE: 23.892573551270335
# MSE: 785.4474108729297
# RMSE: 28.02583470430327
# R2: 0.07783450438798478
# ME: -1.0622016749183203
# MAV: 51.70756402620607
# MPV: 2.158769736292201
# RME: -1.8740993095216731
# RMAE: 2.158769736292201

# Modelo: MLR com forward selection with split date without groupping by days
# MAE: 41.15081159744376
# MSE: 1768.6389494628397
# RMSE: 42.055189328581555
# R2: -457.00198078654336
# ME: 41.15081159744376
# MAV: 86.73989898989899
# MPV: 0.47362513807015366
# RME: 0.47362513807015366
# RMAE: 0.47362513807015366

# Modelo: MLR com forward selection with groupping no shuffle = no groupping no shuffle
# MAE: 45.00000000000001
# MSE: 2051.9166666666674
# RMSE: 45.29808678814887
# R2: -75.23219814241489
# ME: 45.00000000000001
# MAV: 81.5
# MPV: 0.5503190889614168
# RME: 0.5503190889614168
# RMAE: 0.5503190889614168

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

# Modelo: Lasso with random split and groupping
# MAE: 46.659160539185585
# MSE: 2194.321817141358
# RMSE: 46.84358885846982
# R2: -547.5804542853396
# ME: 46.659160539185585
# MAV: 87.0
# MPV: 0.5357370022142913
# RME: 0.5357370022142913
# RMAE: 0.5357370022142913

# Modelo: Lasso with split date and groupping
# MAE: 46.659160539185585
# MSE: 2194.321817141358
# RMSE: 46.84358885846982
# R2: -547.5804542853396
# ME: 46.659160539185585
# MAV: 87.0
# MPV: 0.5357370022142913
# RME: 0.5357370022142913
# RMAE: 0.5357370022142913

# Modelo: Lasso with random split and without groupping
# MAE: 46.659160539185585
# MSE: 2194.321817141358
# RMSE: 46.84358885846982
# R2: -547.5804542853396
# ME: 46.659160539185585
# MAV: 87.0
# MPV: 0.5357370022142913
# RME: 0.5357370022142913
# RMAE: 0.5357370022142913

# Modelo: Lasso with split date and without groupping
# MAE: 46.659160539185585
# MSE: 2194.321817141358
# RMSE: 46.84358885846982
# R2: -547.5804542853396
# ME: 46.659160539185585
# MAV: 87.0
# MPV: 0.5357370022142913
# RME: 0.5357370022142913
# RMAE: 0.5357370022142913

# Modelo: Lasso no groupping no shuffle
# MAE: 45.31689027135299
# MSE: 2090.6754545230656
# RMSE: 45.72390462901288
# R2: -76.67215310921605
# ME: 45.31689027135299
# MAV: 81.5
# MPV: 0.5536739795478205
# RME: 0.5536739795478205
# RMAE: 0.5536739795478205


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
# MAE: 22.479075310607097
# MSE: 745.6133734857762
# RMSE: 27.305921949016412
# R2: -0.1643863683919904
# ME: -8.430637207921599
# MAV: 39.388888888888886
# MPV: 3.8954682689765527
# RME: -3.7082382248189685
# RMAE: 3.8954682689765527

# Modelo: Ridge with split date and groupping
# MAE: 46.233275403437865
# MSE: 2202.3435517933417
# RMSE: 46.929133294717275
# R2: -549.5858879483354
# ME: 46.233275403437865
# MAV: 87.0
# MPV: 0.5299853644154333
# RME: 0.5299853644154333
# RMAE: 0.5299853644154333

# Modelo: Ridge with random split and without groupping
# MAE: 22.479075310607097
# MSE: 745.6133734857762
# RMSE: 27.305921949016412
# R2: -0.1643863683919904
# ME: -8.430637207921599
# MAV: 39.388888888888886
# MPV: 3.8954682689765527
# RME: -3.7082382248189685
# RMAE: 3.8954682689765527

# Modelo: Ridge with split date and without groupping
# MAE: 46.233275403437865
# MSE: 2202.3435517933417
# RMSE: 46.929133294717275
# R2: -549.5858879483354
# ME: 46.233275403437865
# MAV: 87.0
# MPV: 0.5299853644154333
# RME: 0.5299853644154333
# RMAE: 0.5299853644154333

# Modelo: Ridge no groupping, no shuffle
# MAE: 47.173319994266336
# MSE: 2282.9084380819895
# RMSE: 47.779791105466224
# R2: -83.81393578013582
# ME: 47.173319994266336
# MAV: 81.5
# MPV: 0.575905969107614
# RME: 0.575905969107614
# RMAE: 0.575905969107614


####################### GLM gaussian ######################

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

# Modelo: GLM com Gaussian with random split groupping by days
# MAE: 22.487137084972776
# MSE: 754.8074025844634
# RMSE: 27.47375843572305
# R2: -0.1787442146080025
# ME: -8.48251293223039
# MAV: 39.388888888888886
# MPV: 3.8968424645246
# RME: -3.71025481212987
# RMAE: 3.8968424645246

# Modelo: GLM com Gaussian with split date groupping by days
# MAE: 48.3129447181171
# MSE: 2370.913633854253
# RMSE: 48.692028442592665
# R2: -591.7284084635633
# ME: 48.3129447181171
# MAV: 87.0
# MPV: 0.5545537763055371
# RME: 0.5545537763055371
# RMAE: 0.5545537763055371

# Modelo: GLM com Gaussian with random split without groupping by days
# MAE: 23.892573551270335
# MSE: 785.4474108729297
# RMSE: 28.02583470430327
# R2: 0.07783450438798478
# ME: -1.0622016749183203
# MAV: 51.70756402620607
# MPV: 2.158769736292201
# RME: -1.8740993095216731
# RMAE: 2.158769736292201

# Modelo: GLM com Gaussian with split date without groupping by days
# MAE: 41.15081159744376
# MSE: 1768.6389494628397
# RMSE: 42.055189328581555
# R2: -457.00198078654336
# ME: 41.15081159744376
# MAV: 86.73989898989899
# MPV: 0.47362513807015366
# RME: 0.47362513807015366
# RMAE: 0.47362513807015366

# Modelo: GLM com Gaussian no groupping no shuffle
# MAE: 45.00000000000001
# MSE: 2051.9166666666674
# RMSE: 45.29808678814887
# R2: -75.23219814241489
# ME: 45.00000000000001
# MAV: 81.5
# MPV: 0.5503190889614168
# RME: 0.5503190889614168
# RMAE: 0.5503190889614168


####################### GLM kernel ######################

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
# MAE: 25.86538746742178
# MSE: 987.8904678897654
# RMSE: 31.43072490239074
# R2: -0.542738147114487
# ME: -7.668804056266186
# MAV: 39.388888888888886
# MPV: 4.585017642904079
# RME: -4.328950327666964
# RMAE: 4.585017642904079

# Modelo: GLM com Kernel Polinomial with split date groupping by days
# MAE: 50.1745672028449 
# MSE: 2949.244571332167
# RMSE: 54.30694772616269
# R2: -736.3111428330418
# ME: 50.1745672028449
# MAV: 87.0
# MPV: 0.5743785017264725
# RME: 0.5743785017264725
# RMAE: 0.5743785017264725

# Modelo: GLM com Kernel Polinomial with random split without groupping by days
# MAE: 22.52204920866659
# MSE: 735.0911984534324
# RMSE: 27.112565324097098
# R2: 0.1369559184255723
# ME: -1.0578395931646678
# MAV: 51.70756402620607
# MPV: 2.034512273145512
# RME: -1.7603256365908797
# RMAE: 2.034512273145512

# Modelo: GLM com Kernel Polinomial with split date without groupping by days
# MAE: 38.67524890753548
# MSE: 1621.9610971866741
# RMSE: 40.27357815226596
# R2: -419.01867905025455
# ME: 38.658310361071926
# MAV: 86.73989898989899
# MPV: 0.44500553485281563
# RME: 0.44480388549015426
# RMAE: 0.44500553485281563

# Modelo: GLM com Kernel Polinomial no groupping no shuffle
# MAE: 65.25130531736362
# MSE: 4871.181549918622
# RMSE: 69.79385037321427
# R2: -179.97268916106336
# ME: 28.223615836688708
# MAV: 81.5
# MPV: 0.8023156753977738
# RME: 0.3252694309182104
# RMAE: 0.8023156753977738


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
# X_train = train.drop(columns=['Date', 'Screening_date'])
# y_train = train['Date']

# X_test = test.drop(columns=['Date', 'Screening_date'])
# y_test = test['Date']

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

# Modelo: SVR with random split groupping by days
# MAE: 23.730641547378116
# MSE: 759.6626559797085
# RMSE: 27.561978448212102
# R2: -0.18632641614776624
# ME: -9.378516430704996
# MAV: 39.388888888888886
# MPV: 3.9366663526218266
# RME: -3.74422740796525
# RMAE: 3.9366663526218266

# Modelo: SVR with split date groupping by days
# MAE: 45.1020912873053
# MSE: 2038.2190229808348
# RMSE: 45.146639110578704
# R2: -508.5547557452087
# ME: 45.1020912873053
# MAV: 87.0
# MPV: 0.5181587866120185
# RME: 0.5181587866120185
# RMAE: 0.5181587866120185

# Modelo: SVR with random split without groupping by days
# MAE: 22.11268100873779
# MSE: 791.5232586981931
# RMSE: 28.134023151660926
# R2: 0.07070107044513141
# ME: -5.9272512830516195
# MAV: 51.70756402620607
# MPV: 2.303801638986002
# RME: -2.1000095180576928
# RMAE: 2.303801638986002

# Modelo: SVR with split date without groupping by days
# MAE: 37.221691233479284
# MSE: 1390.3897503648172
# RMSE: 37.28793035775541
# R2: -359.0515865184351
# ME: 37.221691233479284
# MAV: 86.73989898989899
# MPV: 0.42881856698570986
# RME: 0.42881856698570986
# RMAE: 0.42881856698570986

# Modelo: SVR no groupping no shuffle
# MAE: 45.84273368142536
# MSE: 2131.442603423642
# RMSE: 46.16754924645277
# R2: -78.18672210861828
# ME: 45.84273368142536
# MAV: 81.5
# MPV: 0.5605298227180829
# RME: 0.5605298227180829
# RMAE: 0.5605298227180829


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


#### results
# Modelo: SVR group and no shuffle
# MAE: 48.40969019413729
# MSE: 2400.6367310350047
# RMSE: 48.99629303360617
# R2: -88.18774232947385
# ME: 48.40969019413729
# MAV: 81.5
# MPV: 0.5909392002773495
# RME: 0.5909392002773495
# RMAE: 0.5909392002773495

# Modelo: SVR group and split date
# MAE: 47.52717770034388
# MSE: 2280.867244836854
# RMSE: 47.75842590409418
# R2: -569.2168112092135
# ME: 47.52717770034388
# MAV: 87.0
# MPV: 0.5454792677833197
# RME: 0.5454792677833197
# RMAE: 0.5454792677833197

# Modelo: SVR no group and split date
# MAE: 37.8783317634262
# MSE: 1644.2367723421753
# RMSE: 40.54918953989309
# R2: -424.78712791748984
# ME: 37.8783317634262
# MAV: 86.73989898989899
# MPV: 0.4354852221561902
# RME: 0.4354852221561902
# RMAE: 0.4354852221561902

# Modelo: SVR no group and no shuffle
# MAE: 39.53445575208402
# MSE: 1804.0413037511855
# RMSE: 42.47400738982826
# R2: -409.96586391628847
# ME: 39.53445575208402
# MAV: 86.52829064919595
# MPV: 0.4561120271731581
# RME: 0.4561120271731581
# RMAE: 0.4561120271731581