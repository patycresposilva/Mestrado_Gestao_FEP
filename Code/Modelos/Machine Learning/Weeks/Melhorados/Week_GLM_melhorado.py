# Week_GLM_melhorado

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

# Listas de colunas por tipo
categorical_columns = ['Gender', 'Personal_cancer_history', 'Family_cancer_history',
                       'Sun_exposure', 'Body_signs', 'Premalignant', 'Malignant',
                       'Phototype_2', 'Phototype_3', 'Phototype_4', 'Phototype_5',
                       'Phototype_6', 'day_of_week', 'month', 'is_weekend', 'quarter']

numerical_columns = ['Age', 'Total_screenings', 'year', 'day_of_year', 'week_of_year']

# Corrigir os tipos de dados
df['Screening_date'] = pd.to_datetime(df['Screening_date'])  # Converter para datetime

# Converter as colunas categóricas para o tipo category
df[categorical_columns] = df[categorical_columns].astype('category')

# Converter todas as colunas numéricas para int64
df[numerical_columns] = df[numerical_columns].astype('int64')

# Verificar os tipos de dados após a correção
print(df.dtypes)

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
X = df.drop(columns=['Total_screenings', 'day_of_year', 'quarter', 'month'])
y = df['Total_screenings']

# # Definir as features (X) e a variável target (y)
# X = df.drop(columns=['Total_screenings'])
# y = df['Total_screenings']

print(X.columns)
print(y.head(75))

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# print(X_train.shape, X_test.shape)

####################### GLM ######################

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Gaussian
from sklearn.preprocessing import PolynomialFeatures

# Adicionar uma constante aos dados
X_train_sm = sm.add_constant(X_train)
X_test_sm = X_test.copy()  # Certifique-se de que não está sobrescrevendo X_test original
X_test_sm.insert(0, 'const', 1.0)

print(X_train_sm.head())
print(X_test_sm.head())

### Gaussian ###

# Ajustar o modelo GLM com família Gaussian
model_glm = GLM(y_train, X_train_sm, family=Gaussian()).fit()

# Obter o resumo do modelo
print(model_glm.summary())

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

print("Modelo: GLM com Gaussian com todas as vvs")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# Modelo: GLM com Gaussian todas as vvs
# MAE: 147.1955513224075
# MSE: 27448.460949768993
# RMSE: 165.67577055734188
# R2: -2.61603805429881
# ME: 142.78331943574415
# MAV: 187.66666666666666
# MPV: 0.7907785111781443
# RME: 0.6386325840518217
# RMAE: 0.7907785111781443

### features importance - maior coeficiente, menor p-value

# Obter o resumo do modelo
print(model_glm.summary())

# # Remover variáveis não significativas
# X_train_sm_reduced = X_train_sm.drop(['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Body_signs', 
#                                       'Phototype_2', 'Phototype_3', 'Phototype_4', 'month', 
#                                       'is_weekend', 'quarter', 'Age'], axis=1)
# X_test_sm_reduced = X_test_sm.drop(['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Body_signs', 
#                                     'Phototype_2', 'Phototype_3', 'Phototype_4', 'month', 
#                                     'is_weekend', 'quarter', 'Age'], axis=1)

# Remover variáveis não significativas
X_train_sm_reduced = X_train_sm.drop(['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Body_signs', 
                                      'Phototype_2', 'Phototype_3', 'Phototype_4', 
                                      'is_weekend', 'Age'], axis=1)
X_test_sm_reduced = X_test_sm.drop(['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Body_signs', 
                                    'Phototype_2', 'Phototype_3', 'Phototype_4', 
                                    'is_weekend', 'Age'], axis=1)

# Ajustar o modelo novamente com as variáveis reduzidas
model_glm_reduced = sm.OLS(y_train, X_train_sm_reduced).fit()

# Obter o resumo do modelo reduzido
print(model_glm_reduced.summary())

# Fazer previsões no conjunto de teste
y_pred_reduced = model_glm_reduced.predict(X_test_sm_reduced)

# Calcular e imprimir as métricas de avaliação
mae = mean_absolute_error(y_test, y_pred_reduced)
mse = mean_squared_error(y_test, y_pred_reduced)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_reduced)
me = np.mean(y_test - y_pred_reduced)
mav = np.mean(np.abs(y_test))
mpv = np.mean(np.abs((y_test - y_pred_reduced) / y_test))
rme = np.mean((y_test - y_pred_reduced) / y_test)
rmae = np.mean(np.abs(y_test - y_pred_reduced) / np.abs(y_test))

print("Modelo: GLM com apenas as vvs importantes")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# Modelo: GLM com apenas as vvs importantes
# MAE: 148.72429989811624
# MSE: 28232.82092436453
# RMSE: 168.02625070019425
# R2: -2.7193690032214732
# ME: 143.41914156927308
# MAV: 187.66666666666666
# MPV: 0.798932282971084
# RME: 0.6159957888730444
# RMAE: 0.798932282971084


# Selecionar apenas as colunas numéricas
numeric_columns = X_train_sm.select_dtypes(include=[np.number]).columns

# Calcular o Z-Score para as variáveis numéricas
z_scores = np.abs((X_train_sm[numeric_columns] - X_train_sm[numeric_columns].mean()) / X_train_sm[numeric_columns].std())

# Definir um limiar para identificar outliers
threshold = 3

# Identificar os outliers
outliers = (z_scores > threshold).any(axis=1)

# Remover os outliers
X_train_filtered = X_train_sm[~outliers]
y_train_filtered = y_train[~outliers]

print(f"Outliers detectados e removidos: {outliers.sum()}")

# Agora ajuste o modelo com os dados sem outliers
model_glm = GLM(y_train_filtered, X_train_filtered, family=Gaussian()).fit()

# Obter o resumo do modelo reduzido
print(model_glm.summary())

# Fazer previsões no conjunto de teste
y_pred_reduced = model_glm.predict(X_test_sm)

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

print("Modelo: GLM sem outliers")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# Modelo: GLM sem outliers
# MAE: 147.1955513224075
# MSE: 27448.460949768993
# RMSE: 165.67577055734188
# R2: -2.61603805429881
# ME: 142.78331943574415
# MAV: 187.66666666666666
# MPV: 0.7907785111781443
# RME: 0.6386325840518217
# RMAE: 0.7907785111781443

