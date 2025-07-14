# Week_MLP NN_melhorado

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

# # Converter as colunas categóricas para o tipo category
# df[categorical_columns] = df[categorical_columns].astype('category')

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

# # Definir as features (X) e a variável target (y)
# X = df.drop(columns=['Total_screenings', 'day_of_year', 'quarter', 'month'])
# y = df['Total_screenings']

# Definir as features (X) e a variável target (y)
X = df.drop(columns=['Total_screenings'])
y = df['Total_screenings']

print(X.columns)
print(y.head(75))

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# print(X_train.shape, X_test.shape)

####################### MLP NN: Multilayer Perceptron Neural Network ######################

from sklearn.neural_network import MLPRegressor

# Definir o modelo MLP com uma camada oculta de 100 neurônios (você pode ajustar os parâmetros conforme necessário)
model_mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Ajustar o modelo aos dados de treino
model_mlp.fit(X_train, y_train)

# "Summary" do modelo
print(f"Coefs: {model_mlp.coefs_}")
print(f"Intercepts: {model_mlp.intercepts_}")
print(f"Número de iterações: {model_mlp.n_iter_}")

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

print("Modelo: MPL NN")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# Modelo: MPL NN com todas as vvs
# MAE: 81.40105419883837
# MSE: 10022.494344726054
# RMSE: 100.11240854522507
# R2: -0.3203553021004275
# ME: 49.72361663794801
# MAV: 187.66666666666666
# MPV: 0.6341063900136015
# RME: -0.09718300235266589
# RMAE: 0.6341063900136015

## features importance

# Obtenha os coeficientes da primeira camada
first_layer_weights = model_mlp.coefs_[0]

# Calcule a importância das variáveis como a soma dos valores absolutos dos pesos de cada neurônio da primeira camada
importance = np.sum(np.abs(first_layer_weights), axis=1)

# Ordene as variáveis por importância
indices = np.argsort(importance)[::-1]

print("Importância das variáveis:")
for i in indices:
    print(f"Variável {i}: {importance[i]}")

# Índices de variáveis a serem mantidas (importância >= 10)
selected_indices = [i for i, imp in enumerate(importance) if imp >= 10.7]

# Reduzindo o conjunto de dados apenas às variáveis selecionadas
X_train_reduced = X_train.iloc[:, selected_indices]
X_test_reduced = X_test.iloc[:, selected_indices]

# Criando e ajustando o novo modelo MLP com as variáveis selecionadas
model_mlp_reduced = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
model_mlp_reduced.fit(X_train_reduced, y_train)

# Exibindo o número de variáveis após a redução e o novo modelo ajustado
num_vars_before = X_train.shape[1]
num_vars_after = X_train_reduced.shape[1]

print(f"Número de variáveis antes da redução: {num_vars_before}")
print(f"Número de variáveis após a redução: {num_vars_after}")
print(f"Modelo ajustado: {model_mlp_reduced}")

# Fazer previsões no conjunto de teste
y_pred_reduced = model_mlp_reduced.predict(X_test_reduced)

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

print("Modelo: MPL NN com features importance")
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# Modelo: MPL NN com features importance > 10
# MAE: 99.74667550458827
# MSE: 14290.298186036038
# RMSE: 119.54203522625853
# R2: -0.8825923297684368
# ME: 68.64124461755999
# MAV: 187.66666666666666
# MPV: 0.8032687275040363
# RME: -0.08814380876475376
# RMAE: 0.8032687275040363

# Modelo: MPL NN com features importance > 10.7
# MAE: 106.52076138749455
# MSE: 15980.684023270169
# RMSE: 126.41473024640035
# R2: -1.1052823933414841
# ME: 93.04865275657446
# MAV: 187.66666666666666
# MPV: 0.6264426365749726
# RME: 0.24242675297704078
# RMAE: 0.6264426365749726

## remove outliers

from scipy import stats

# retirar transformação de vvs para categóricas
# Calcular o Z-score para cada variável no conjunto de treino
z_scores = np.abs(stats.zscore(X_train))

# Definir o limiar para considerar um valor como outlier
threshold = 3

# Identificar os índices dos outliers
outliers = np.where(z_scores > threshold)
print(outliers)

# Remover os outliers dos dados de treino
X_train_clean = X_train[(z_scores > threshold).any(axis=1)]
y_train_clean = y_train[(z_scores > threshold).any(axis=1)]

# Verificar o número de amostras após a remoção dos outliers
num_samples_X = len(X_train_clean)
num_samples_y = len(y_train_clean)

# Exibir os resultados
print(f"Número de amostras no X_train_clean: {num_samples_X}")
print(f"Número de amostras no y_train_clean: {num_samples_y}")

# Verificar a quantidade de dados após a remoção dos outliers
print(f"Dados originais: {X_train.shape[0]}")
print(f"Dados sem outliers: {X_train_clean.shape[0]}")

# Definir o modelo MLP com os mesmos parâmetros
model_mlp_clean = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Ajustar o modelo aos dados de treino sem outliers
model_mlp_clean.fit(X_train_clean, y_train_clean)

# Fazer previsões no conjunto de teste
y_pred_clean = model_mlp_clean.predict(X_test)

# Calcular e imprimir as métricas de avaliação
mae_clean = mean_absolute_error(y_test, y_pred_clean)
mse_clean = mean_squared_error(y_test, y_pred_clean)
rmse_clean = np.sqrt(mse_clean)
r2_clean = r2_score(y_test, y_pred_clean)
me = np.mean(y_test - y_pred_clean)
mav = np.mean(np.abs(y_test))
mpv = np.mean(np.abs((y_test - y_pred_clean) / y_test))
rme = np.mean((y_test - y_pred_clean) / y_test)
rmae = np.mean(np.abs(y_test - y_pred_clean) / np.abs(y_test))

print("Modelo: MLP NN sem outliers")
print(f'MAE: {mae_clean}')
print(f'MSE: {mse_clean}')
print(f'RMSE: {rmse_clean}')
print(f'R2: {r2_clean}')
print(f'ME: {me}')
print(f'MAV: {mav}')
print(f'MPV: {mpv}')
print(f'RME: {rme}')
print(f'RMAE: {rmae}')

# # Modelo: MLP NN sem outliers
# # MAE: 92.97910112374369
# # MSE: 12334.13772953612
# # RMSE: 111.05916319483107
# # R2: -0.6248893327237968
# # ME: 69.78068500402638
# # MAV: 187.66666666666666
# # MPV: 0.619437715316883
# # RME: 0.06973010698472537
# # RMAE: 0.619437715316883

