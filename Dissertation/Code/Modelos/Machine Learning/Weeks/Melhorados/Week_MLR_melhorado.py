# Week_MLR_melhorado

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

####################### MLR ######################
from sklearn.linear_model import LinearRegression

# Adicionar uma constante aos dados
X_train_sm = sm.add_constant(X_train)
X_test_sm = X_test.copy()  # Certifique-se de que não está sobrescrevendo X_test original
X_test_sm.insert(0, 'const', 1.0)

print(X_train_sm.head())
print(X_test_sm.head())

# Ajustar o modelo
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Obter o resumo do modelo
print(model_sm.summary())

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

# LR with all features
# MAE: 147.19555132227882
# MSE: 27448.460949724315
# RMSE: 165.67577055720704
# R2: -2.616038054292924
# ME: 142.78331943558538
# MAV: 187.66666666666666
# MPV: 0.7907785111778013
# RME: 0.6386325840504415
# RMAE: 0.7907785111778013

### features importance - maior coeficiente, menor p-value

# Obter o resumo do modelo
print(model_sm.summary())

# Remover variáveis não significativas
X_train_sm_reduced = X_train_sm.drop(['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Body_signs', 
                                      'Phototype_2', 'Phototype_3', 'Phototype_4', 'month', 
                                      'is_weekend', 'quarter', 'Age'], axis=1)
X_test_sm_reduced = X_test_sm.drop(['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Body_signs', 
                                    'Phototype_2', 'Phototype_3', 'Phototype_4', 'month', 
                                    'is_weekend', 'quarter', 'Age'], axis=1)


# Ajustar o modelo novamente com as variáveis reduzidas
model_sm_reduced = sm.OLS(y_train, X_train_sm_reduced).fit()

# Obter o resumo do modelo reduzido
print(model_sm_reduced.summary())

# Fazer previsões no conjunto de teste
y_pred_reduced = model_sm_reduced.predict(X_test_sm_reduced)

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

# LR with just important features
# MAE: 148.7242998980619
# MSE: 28232.820924347496
# RMSE: 168.02625070014358
# R2: -2.7193690032192297
# ME: 143.4191415692139
# MAV: 187.66666666666666
# MPV: 0.7989322829708204
# RME: 0.6159957888726135
# RMAE: 0.7989322829708204

### remove outliers
import numpy as np
import statsmodels.api as sm

# Calcular os resíduos
residuos = model_sm.resid

# Calcular os resíduos padronizados
residuos_padronizados = residuos / np.std(residuos)

# Definir um limite para identificar outliers (exemplo: 3 desvios padrão)
limite_superior = 3
limite_inferior = -3

# Identificar os índices dos outliers
outliers = np.where((residuos_padronizados > limite_superior) | (residuos_padronizados < limite_inferior))

print(f'Outliers identificados nos índices: {outliers}')

# Remover os outliers dos conjuntos de treinamento e teste
X_train_sm_sem_outliers = X_train_sm.drop(X_train_sm.index[outliers])
y_train_sem_outliers = y_train.drop(y_train.index[outliers])

# Ajustar o modelo novamente sem os outliers
model_sm_sem_outliers = sm.OLS(y_train_sem_outliers, X_train_sm_sem_outliers).fit()

# Obter o resumo do novo modelo
print(model_sm_sem_outliers.summary())

# Fazer previsões no conjunto de teste original (sem remover outliers do conjunto de teste)
y_pred_sem_outliers = model_sm_sem_outliers.predict(X_test_sm)

# Calcular e imprimir as métricas de avaliação para o modelo sem outliers
mae_sem_outliers = mean_absolute_error(y_test, y_pred_sem_outliers)
mse_sem_outliers = mean_squared_error(y_test, y_pred_sem_outliers)
rmse_sem_outliers = np.sqrt(mse_sem_outliers)
r2_sem_outliers = r2_score(y_test, y_pred_sem_outliers)

print("Modelo Sem Outliers: LR")
print(f'MAE: {mae_sem_outliers}')
print(f'MSE: {mse_sem_outliers}')
print(f'RMSE: {rmse_sem_outliers}')
print(f'R2: {r2_sem_outliers}')

# Modelo Sem Outliers: LR
# MAE: 159.34306733424333
# MSE: 32388.313146934015
# RMSE: 179.96753359129534
# R2: -3.266810189037047

# Lista das variáveis não significativas a serem removidas
variaveis_para_remover = ['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Body_signs', 
                          'Phototype_2', 'Phototype_3', 'Phototype_4', 'month', 
                          'is_weekend', 'quarter', 'Age']

# Remover essas variáveis do conjunto de treino sem outliers
X_train_sm_reduced = X_train_sm_sem_outliers.drop(variaveis_para_remover, axis=1)

# Ajustar o modelo novamente sem as variáveis não significativas
model_sm_reduced2 = sm.OLS(y_train_sem_outliers, X_train_sm_reduced).fit()

# Obter o resumo do novo modelo
print(model_sm_reduced2.summary())

# Fazer previsões no conjunto de teste com o modelo reduzido
X_test_sm_reduced = X_test_sm.drop(variaveis_para_remover, axis=1)
y_pred_reduced = model_sm_reduced2.predict(X_test_sm_reduced)

# Calcular e imprimir as métricas de avaliação para o modelo reduzido
mae_reduced = mean_absolute_error(y_test, y_pred_reduced)
mse_reduced = mean_squared_error(y_test, y_pred_reduced)
rmse_reduced = np.sqrt(mse_reduced)
r2_reduced = r2_score(y_test, y_pred_reduced)

print("Modelo Sem Outliers e Sem Variáveis Não Significativas: LR")
print(f'MAE: {mae_reduced}')
print(f'MSE: {mse_reduced}')
print(f'RMSE: {rmse_reduced}')
print(f'R2: {r2_reduced}')

# MAE: 163.04986595815515
# MSE: 33030.34725955026
# RMSE: 181.74253013411655
# R2: -3.3513912439685747


### remove outliers and non important variables and correlated variables

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Remover as variáveis indesejadas do conjunto de treinamento e teste
variaveis_para_remover = ['day_of_year', 'quarter', 'year', 'is_weekend', 'Premalignant', 'Malignant', 'Phototype_5', 'Phototype_6']

X_train_sm_sem_vars = X_train_sm.drop(columns=variaveis_para_remover)
X_test_sm_sem_vars = X_test_sm.drop(columns=variaveis_para_remover)

# Ajustar o modelo com as variáveis restantes
model_sm = sm.OLS(y_train, X_train_sm_sem_vars).fit()

# Calcular os resíduos
residuos = model_sm.resid

# Calcular os resíduos padronizados
residuos_padronizados = residuos / np.std(residuos)

# Definir um limite para identificar outliers (exemplo: 3 desvios padrão)
limite_superior = 3
limite_inferior = -3

# Identificar os índices dos outliers
outliers = np.where((residuos_padronizados > limite_superior) | (residuos_padronizados < limite_inferior))

print(f'Outliers identificados nos índices: {outliers}')

# Remover os outliers dos conjuntos de treinamento
X_train_sm_sem_outliers = X_train_sm_sem_vars.drop(X_train_sm_sem_vars.index[outliers])
y_train_sem_outliers = y_train.drop(y_train.index[outliers])

# Ajustar o modelo novamente sem os outliers
model_sm_sem_outliers = sm.OLS(y_train_sem_outliers, X_train_sm_sem_outliers).fit()

# Obter o resumo do novo modelo
print(model_sm_sem_outliers.summary())

# Fazer previsões no conjunto de teste original (sem remover outliers do conjunto de teste)
y_pred_sem_outliers = model_sm_sem_outliers.predict(X_test_sm_sem_vars)

# Calcular e imprimir as métricas de avaliação para o modelo sem outliers
mae_sem_outliers = mean_absolute_error(y_test, y_pred_sem_outliers)
mse_sem_outliers = mean_squared_error(y_test, y_pred_sem_outliers)
rmse_sem_outliers = np.sqrt(mse_sem_outliers)
r2_sem_outliers = r2_score(y_test, y_pred_sem_outliers)

print(f'MAE (sem outliers): {mae_sem_outliers}')
print(f'MSE (sem outliers): {mse_sem_outliers}')
print(f'RMSE (sem outliers): {rmse_sem_outliers}')
print(f'R² (sem outliers): {r2_sem_outliers}')

# MAE (sem outliers e sem vv correlacionadas): 128.3922030448043
# MSE (sem outliers e sem vv correlacionadas): 22117.412699126966
# RMSE (sem outliers e sem vv correlacionadas): 148.7192411866298
# R² (sem outliers e sem vv correlacionadas): -1.9137300677453086
