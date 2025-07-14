# week

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

######## no groupping - todas as linhas de cada semana
# Adicionar uma coluna 'Week' que extrai o ano e semana do 'Screening_date'
df['Week'] = df['Screening_date'].dt.strftime('%Y-%U')

# Adicionar coluna com o número total de screenings por semana
df['Total_screenings'] = df.groupby('Week')['Screening_date'].transform('count')

# Mover a coluna 'Total_screenings' para a primeira posição
columns = ['Total_screenings'] + [col for col in df.columns if col != 'Total_screenings']
df = df[columns]

print(df[['Screening_date', 'Week', 'Total_screenings']])

# Eliminar a coluna 'Week'
df = df.drop(columns=['Week'])

## adicionar novas vvs

# Criar novas variáveis derivadas de 'Screening_date'
df['day_of_week'] = df['Screening_date'].dt.dayofweek        # Dia da semana (0 = segunda-feira, 6 = domingo)
df['month'] = df['Screening_date'].dt.month                  # Mês (1 = janeiro, 12 = dezembro)
df['year'] = df['Screening_date'].dt.year                    # Ano
df['day_of_year'] = df['Screening_date'].dt.dayofyear        # Dia do ano (1 a 365 ou 366)
df['week_of_year'] = df['Screening_date'].dt.isocalendar().week  # Semana do ano (1 a 53)
df['is_weekend'] = (df['Screening_date'].dt.dayofweek >= 5).astype(int)  # Se é fim de semana (0 = não, 1 = sim)
df['quarter'] = df['Screening_date'].dt.quarter              # Trimestre (1 a 4)

# Verificar as novas variáveis criadas
print(df[['Screening_date', 'day_of_week', 'month', 'year', 'day_of_year', 'week_of_year', 'is_weekend', 'quarter']])

print(df)
print(df.columns)

## corrigir tipo de vvs
# Verificar os tipos de dados atuais
print("Tipos de dados antes da correção:")
print(df.dtypes)

# Corrigir os tipos de dados
df['Screening_date'] = pd.to_datetime(df['Screening_date'])  # Converter para datetime

print(df)

# ### groupping - uma linha por semana
# Listas de colunas por tipo
categorical_columns = ['Gender', 'Personal_cancer_history', 'Family_cancer_history',
                       'Sun_exposure', 'Body_signs', 'Premalignant', 'Malignant',
                       'Phototype_2', 'Phototype_3', 'Phototype_4', 'Phototype_5',
                       'Phototype_6', 'day_of_week', 'month', 'is_weekend', 'quarter']

# Função personalizada para calcular a moda
def calculate_mode(series):
    return series.mode()[0] if not series.mode().empty else series.iloc[0]

# Definir as funções de agregação para todas as colunas
aggregation_functions = {col: calculate_mode for col in categorical_columns}
aggregation_functions.update({
    'Age': 'median',
    'Total_screenings': 'first',  # Manter o primeiro valor, pois já contém o total da semana
    'year': 'first',              # Manter o primeiro valor
    'day_of_year': 'median',
    'week_of_year': 'first'
})

# Agrupar os dados por 'year' e 'week_of_year', mantendo as colunas no índice
df = df.groupby(['year', 'week_of_year'], as_index=False).agg(aggregation_functions)

# Verificar o resultado do agrupamento
print(df)

print(df.columns)


# # correlação de pearson
# correlation_matrix = df.corr(method='pearson')
# print(correlation_matrix)

# import seaborn as sns
# import matplotlib.pyplot as plt

# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.show()


# Definir as features (X) e a variável target (y)
X = df.drop(columns=['Total_screenings'])
y = df['Total_screenings']

print(X.columns)
print(y.head(75))

# # Dividir os dados em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Definindo as datas de corte
split_date = 2024

# Dividindo os dados com base na data
train = df[df['year'] < split_date]
test = df[df['year'] >= split_date]

# Separando X e y para cada conjunto
X_train = train.drop(columns=['Total_screenings'])
y_train = train['Total_screenings']

X_test = test.drop(columns=['Total_screenings'])
y_test = test['Total_screenings']

# print(X_train.shape, X_test.shape)

# ####################### XGB ######################

import xgboost as xgb

# Instanciar o modelo XGBRegressor
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

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

print("Modelo: XGB com todas as features")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# Modelo: XGB com todas as features
# MAE: 133.57329203287762
# MSE: 23946.566600109647
# RMSE: 154.74678219630172
# R2: -2.154700756072998
# ME: 125.6927256266276
# MAV: 187.66666666666666
# MPV: 0.7367754473929113
# RME: 0.4650317782118768
# RMAE: 0.7367754473929113

# Modelo: XGB com todas as features split date
# MAE: 130.41179820469446
# MSE: 23902.679529921406
# RMSE: 154.6049143136188
# R2: -2.8513665199279785
# ME: 130.01031766619002
# MAV: 199.0
# MPV: 0.5900772693521567
# RME: 0.5841731437859146
# RMAE: 0.5900772693521567


# Obter a importância das features
importances = model.feature_importances_

# Criar um DataFrame com a importância das features
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Exibir a importância das features
print(importance_df)

# Plotar a importância das features
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Para exibir as features mais importantes no topo
plt.show()

# Definir um limiar para a importância das features
importance_threshold = 0.01  # Ajuste conforme necessário

# Selecionar features com importância acima do limiar
important_features = importance_df[importance_df['Importance'] > importance_threshold]['Feature']
print(f'Features selecionadas: {important_features.tolist()}')

# Filtrar o DataFrame X para incluir apenas as features importantes
X_important = X[important_features]

# # Dividir os dados em conjunto de treino e teste
# X_train_important, X_test_important, y_train, y_test = train_test_split(X_important, y, test_size=0.2, shuffle=False)

# Definindo as datas de corte
split_date = 2024

# Dividindo os dados com base na data
train = df[df['year'] < split_date]
test = df[df['year'] >= split_date]

# Separando X e y para cada conjunto
X_train_important = train.drop(columns=['Total_screenings'])
y_train = train['Total_screenings']

X_test_important = test.drop(columns=['Total_screenings'])
y_test = test['Total_screenings']

# Treinar o modelo com as features importantes
model_important = xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False)
model_important.fit(X_train_important, y_train)

# Fazer previsões e avaliar o modelo
y_pred_important = model_important.predict(X_test_important)

# Avaliar o modelo
from sklearn.metrics import mean_squared_error, r2_score

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred_important)
mse = mean_squared_error(y_test, y_pred_important)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_important)
me = np.mean(y_test - y_pred_important)
mav = np.mean(np.abs(y_test))
mpv = np.mean(np.abs((y_test - y_pred_important) / y_test))
rme = np.mean((y_test - y_pred_important) / y_test)
rmae = np.mean(np.abs(y_test - y_pred_important) / np.abs(y_test))

# print("Modelo: XGB com features mais importantes")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

# XGB apenas com as features mais importantes no shuffle
# MAE: 134.8724510192871
# MSE: 24045.008989982118
# RMSE: 155.0645316956206
# R2: -2.1676692962646484
# ME: 125.7100700378418
# MAV: 187.66666666666666
# MPV: 0.7487046209169012
# RME: 0.4425001543195973
# RMAE: 0.7487046209169012

# Modelo: XGB com features mais importantes split date
# MAE: 130.41179820469446
# MSE: 23902.679529921406
# RMSE: 154.6049143136188
# R2: -2.8513665199279785
# ME: 130.01031766619002
# MAV: 199.0
# MPV: 0.5900772693521567
# RME: 0.5841731437859146
# RMAE: 0.5900772693521567

# # Plotar valores reais e previsões
# plt.figure(figsize=(14, 7))
# plt.plot(y_test.values, label='Valores Reais', color='blue', linestyle='-', marker='o')
# plt.plot(y_pred_important, label='Previsões', color='red', linestyle='--', marker='x')
# plt.xlabel('Índice')
# plt.ylabel('Total de Screenings')
# plt.title('Valores Reais vs. Previsões do Modelo XGB com Features Importantes')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 7))
# plt.scatter(y_test, y_pred_important, alpha=0.5)
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', color='red')
# plt.xlabel('Valores Reais')
# plt.ylabel('Previsões')
# plt.title('Valores Reais vs. Previsões do Modelo XGB com Features Importantes')
# plt.grid(True)
# plt.show()

from scipy import stats

# # Calcular o Z-Score
# z_scores = stats.zscore(df['Total_screenings'])
# df['z_score'] = z_scores

# # Definir um limite para identificar outliers
# threshold = 3
# outliers_z = df[df['z_score'].abs() > threshold]

# print(f'Número de outliers identificados pelo Z-Score: {outliers_z.shape[0]}')
# print(outliers_z)

# # Calcular Q1 e Q3
# Q1 = df['Total_screenings'].quantile(0.25)
# Q3 = df['Total_screenings'].quantile(0.75)
# IQR = Q3 - Q1

# # Definir limites para identificar outliers
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Filtrar outliers
# df_filtered = df[(df['Total_screenings'] >= lower_bound) & (df['Total_screenings'] <= upper_bound)]

# # Verificar o número de linhas removidas
# print(f"Número de linhas originais: {len(df)}")
# print(f"Número de linhas após remoção de outliers: {len(df_filtered)}")

# # Dividir novamente os dados limpos
# X_clean = df_filtered.drop(columns=['Total_screenings'])
# y_clean = df_filtered['Total_screenings']

# # # Dividir os dados em conjunto de treino e teste
# # X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_clean, y_clean, test_size=0.2, shuffle=False)

# # Definindo as datas de corte
# split_date = 2024

# # Dividindo os dados com base na data
# train = df[df['year'] < split_date]
# test = df[df['year'] >= split_date]

# # Separando X e y para cada conjunto
# X_train_cleaned = train.drop(columns=['Total_screenings'])
# y_train_cleaned = train['Total_screenings']

# X_test_cleaned = test.drop(columns=['Total_screenings'])
# y_test_cleaned = test['Total_screenings']

# # Treinar o modelo com as features importantes
# model_important = xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False)
# model_important.fit(X_train_cleaned, y_train_cleaned)

# # Fazer previsões e avaliar o modelo
# y_pred_cleaned = model_important.predict(X_test_cleaned)

# # Calcular e imprimir as métricas de avaliação
# mae = mean_absolute_error(y_test_cleaned, y_pred_cleaned)
# mse = mean_squared_error(y_test_cleaned, y_pred_cleaned)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test_cleaned, y_pred_cleaned)
# me = np.mean(y_test_cleaned - y_pred_cleaned)
# mav = np.mean(np.abs(y_test_cleaned))
# mpv = np.mean(np.abs((y_test_cleaned - y_pred_cleaned) / y_test_cleaned))
# rme = np.mean((y_test_cleaned - y_pred_cleaned) / y_test_cleaned)
# rmae = np.mean(np.abs(y_test_cleaned - y_pred_cleaned) / np.abs(y_test_cleaned))

# print("Modelo: XGB sem outliers")
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'RMSE: {rmse}')
# print(f'R2: {r2}')
# print(f'ME: {me}')
# print(f'MAV: {mav}')
# print(f'MPV: {mpv}')
# print(f'RME: {rme}')
# print(f'RMAE: {rmae}')

# Modelo: XGB sem outliers shuffle
# MAE: 11.93612200419108
# MSE: 353.52369289645793
# RMSE: 18.80222574315227
# R2: 0.9203085899353027
# ME: 2.221896743774414
# MAV: 167.2
# MPV: 0.06149766251610725
# RME: 0.01417796058201034
# RMAE: 0.06149766251610725

# Modelo: XGB sem outliers split date
# Modelo: XGB sem outliers
# MAE: 12.21353258405413
# MSE: 334.8829183442036
# RMSE: 18.299806511113815
# R2: 0.9460413455963135
# ME: 4.269974844796317
# MAV: 199.0
# MPV: 0.05990296171982858
# RME: 0.01838228061957956
# RMAE: 0.05990296171982858

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy import stats

# 1. Obter a importância das features e selecionar as importantes
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Definir um limiar para a importância das features
importance_threshold = 0.01
important_features = importance_df[importance_df['Importance'] > importance_threshold]['Feature']

# 2. Remover Outliers
# Aplicar apenas para as features importantes
df_important = df[important_features.tolist() + ['Total_screenings']]

# Calcular o Z-Score
z_scores = stats.zscore(df_important[['Total_screenings']])
df_important['z_score'] = z_scores

# Definir um limite para identificar outliers
threshold = 3
df_no_outliers = df_important[df_important['z_score'].abs() <= threshold]

# Alternativamente, você pode usar o método IQR para remover outliers
Q1 = df_important['Total_screenings'].quantile(0.25)
Q3 = df_important['Total_screenings'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df_important[(df_important['Total_screenings'] >= lower_bound) & (df_important['Total_screenings'] <= upper_bound)]

# Remover coluna auxiliar 'z_score'
df_no_outliers = df_no_outliers.drop(columns=['z_score'])

# Dividir os dados em conjunto de treino e teste
X_clean = df_no_outliers.drop(columns=['Total_screenings'])
y_clean = df_no_outliers['Total_screenings']

X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_clean, y_clean, test_size=0.2, shuffle=False)

# 3. Treinar o modelo com as features importantes e sem outliers
model_cleaned = xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False)
model_cleaned.fit(X_train_cleaned, y_train_cleaned)

# Fazer previsões e avaliar o modelo
y_pred_cleaned = model_cleaned.predict(X_test_cleaned)

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test_cleaned, y_pred_cleaned)
mse = mean_squared_error(y_test_cleaned, y_pred_cleaned)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_cleaned, y_pred_cleaned)
me = np.mean(y_test_cleaned - y_pred_cleaned)
mav = np.mean(np.abs(y_test_cleaned))
mpv = np.mean(np.abs((y_test_cleaned - y_pred_cleaned) / y_test_cleaned))
rme = np.mean((y_test_cleaned - y_pred_cleaned) / y_test_cleaned)
rmae = np.mean(np.abs(y_test_cleaned - y_pred_cleaned) / np.abs(y_test_cleaned))

print("Modelo XGB com Features Importantes e Sem Outliers")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# Modelo XGB com Features Importantes e Sem Outliers
# MAE: 119.79650961558023
# MSE: 18832.80000923663
# RMSE: 137.232649210152
# R2: -3.245293140411377
# ME: 111.67532227834066
# MAV: 167.2
# MPV: 0.7428544696126875
# RME: 0.4628135269492536
# RMAE: 0.7428544696126875

# Plotar valores reais e previsões
import matplotlib.pyplot as plt

# plt.figure(figsize=(14, 7))
# plt.plot(y_test_cleaned.values, label='Valores Reais', color='blue', linestyle='-', marker='o')
# plt.plot(y_pred_cleaned, label='Previsões', color='red', linestyle='--', marker='x')
# plt.xlabel('Índice')
# plt.ylabel('Total de Screenings')
# plt.title('Valores Reais vs. Previsões do Modelo XGB com Features Importantes e Sem Outliers')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 7))
# plt.scatter(y_test_cleaned, y_pred_cleaned, alpha=0.5)
# plt.plot([min(y_test_cleaned), max(y_test_cleaned)], [min(y_test_cleaned), max(y_test_cleaned)], 'k--', color='red')
# plt.xlabel('Valores Reais')
# plt.ylabel('Previsões')
# plt.title('Valores Reais vs. Previsões do Modelo XGB com Features Importantes e Sem Outliers')
# plt.grid(True)
# plt.show()
