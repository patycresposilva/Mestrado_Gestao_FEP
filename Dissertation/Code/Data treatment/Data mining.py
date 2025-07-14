# Data mining

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


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

############### HISTOGRAMA ###################

# # Criar histogramas para cada coluna numérica
# df.select_dtypes(include=['int64', 'float64']).hist(figsize=(15, 10), bins=20, edgecolor='black')
# plt.tight_layout()
# plt.show()

# # Criar gráficos de barras para cada coluna categórica
# categorical_columns = df.select_dtypes(include=['category', 'object']).columns

# for column in categorical_columns:
#     plt.figure(figsize=(8, 6))
#     df[column].value_counts().plot(kind='bar', edgecolor='black')
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()

############### MDL NÍVEL INFLUÊNCIA FEATURES NA TARGET ###################


##### TARGET: Screening_date

# # Supondo que você já tenha o DataFrame `df`
# X = df.drop(columns=['Screening_date'])  # Substitua 'target' pela sua variável alvo
# y = df['Screening_date']

# # Se a variável alvo for contínua (regressão)
# mi_scores = mutual_info_regression(X, y)

# # Criar um DataFrame para visualizar os resultados
# mi_df = pd.DataFrame({'Variable': X.columns, 'Mutual Information': mi_scores})
# mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# print(mi_df)

#### results mutual information with 'Screening_date'
#                    Variable  Mutual Information
# 1                       Age            0.402905
# 4              Sun_exposure            0.260274
# 9               Phototype_3            0.194727
# 5                Body_signs            0.119534
# 8               Phototype_2            0.116783
# 10              Phototype_4            0.065368
# 3     Family_cancer_history            0.059454
# 2   Personal_cancer_history            0.041634
# 0                    Gender            0.021196
# 6              Premalignant            0.009899
# 11              Phototype_5            0.003722
# 7                 Malignant            0.000000
# 12              Phototype_6            0.000000


####### TARGET: 1 TO 90

# # Agrupar os dados por dias, usando mediana para numéricas e moda para categóricas
# df = df.groupby('Screening_date').agg(lambda x: x.mode()[0] if x.dtype == 'O' else x.median()).reset_index()

# # Número total de datas diferentes
# num_dates = df['Screening_date'].nunique()
# print(f'Total de datas diferentes: {num_dates}')

# # Criar uma nova coluna numérica contínua baseada nas datas
# df['Date'] = range(1, num_dates + 1)

# # Reordenar as colunas para colocar a nova coluna 'Date' em primeiro lugar
# cols = ['Date'] + [col for col in df.columns if col != 'Date']
# df = df[cols]

# print(df)
# print(df.columns)

# # Número total de datas diferentes
# num_dates = df['Screening_date'].nunique()
# print(f'Total de datas diferentes: {num_dates}')

# # Criar um mapeamento de cada data única para um número contínuo
# date_mapping = {date: i+1 for i, date in enumerate(sorted(df['Screening_date'].unique()))}

# # Criar uma nova coluna 'Date' mapeando 'Screening_date' para números contínuos
# df['Date'] = df['Screening_date'].map(date_mapping)

# # Reordenar as colunas para colocar a nova coluna 'Date' em primeiro lugar
# cols = ['Date'] + [col for col in df.columns if col != 'Date']
# df = df[cols]

# # Definir as features (X) e a variável target (y)
# X = df.drop(columns=['Screening_date', 'Date'])
# y = df['Date']

# print(y)

# # Se a variável alvo for contínua (regressão)
# mi_scores = mutual_info_regression(X, y)

# # Criar um DataFrame para visualizar os resultados
# mi_df = pd.DataFrame({'Variable': X.columns, 'Mutual Information': mi_scores})
# mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# print(mi_df)

#### results mutual information with 'Screening_date' from 1 to 90
#                    Variable  Mutual Information
# 4              Sun_exposure        0.113512 
# 9               Phototype_3        0.095712
# 5                Body_signs        0.045979
# 10              Phototype_4        0.003704
# 3     Family_cancer_history        0.000801
# 7                 Malignant        0.000000000000001110223
# 11              Phototype_5        0.000000000000001110223
# 12              Phototype_6        0.000000000000001110223
# 2   Personal_cancer_history        0.0000000000000004440892
# 0                    Gender        0
# 1                       Age        0
# 6              Premalignant        0
# 8               Phototype_2        0


####### TARGET: Date since reference

# # Definir uma data de referência
# reference_date = pd.Timestamp('2022-01-01')

# # Calcular a diferença em dias a partir da data de referência
# df['Days_from_reference'] = (df['Screening_date'] - reference_date).dt.days

# # Ver os valores únicos da coluna 'Days_from_reference'
# unique_values = df['Days_from_reference'].unique()
# print(unique_values)

# # Agrupar os dados por 'Days_from_reference', usando mediana para numéricas e moda para categóricas
# df = df.groupby('Days_from_reference').agg(
#     lambda x: x.mode()[0] if x.dtype == 'O' or x.dtype.name == 'category' else x.median()
# ).reset_index()

# # Definir as features (X) e a variável target (y)
# X = df.drop(columns=['Screening_date', 'Days_from_reference'])
# y = df['Days_from_reference']

# print(y.head(100))

# # Se a variável alvo for contínua (regressão)
# mi_scores = mutual_info_regression(X, y)

# # Criar um DataFrame para visualizar os resultados
# mi_df = pd.DataFrame({'Variable': X.columns, 'Mutual Information': mi_scores})
# mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# print(mi_df)

#### results mutual information with 'Screening_date' as date since reference
#                    Variable  Mutual Information
# 4              Sun_exposure        1.358439e-01
# 9               Phototype_3        1.174259e-01
# 5                Body_signs        1.850980e-02
# 0                    Gender        4.253679e-03
# 8               Phototype_2        5.499754e-04
# 7                 Malignant        1.110223e-15
# 11              Phototype_5        1.110223e-15
# 12              Phototype_6        1.110223e-15
# 2   Personal_cancer_history        4.440892e-16
# 10              Phototype_4        4.440892e-16
# 1                       Age        0.000000e+00
# 3     Family_cancer_history        0.000000e+00
# 6              Premalignant        0.000000e+00


####### TARGET: Screening_count per day

# # Adicionar coluna com o número total de screenings por dia
# df['Total_screenings'] = df.groupby('Screening_date')['Screening_date'].transform('count')

# lenght_data = len('Total_screenings')
# print({lenght_data})

# # Agrupar os dados por dias, usando mediana para numéricas e moda para categóricas
# aggregations = {col: (lambda x: x.mode()[0] if x.dtype == 'O' else x.median()) for col in df.columns if col not in ['Screening_date', 'Total_screenings']}
# aggregations['Total_screenings'] = 'first'

# df = df.groupby('Screening_date').agg(aggregations).reset_index()

# print(df)

# # Definir as features (X) e a variável target (y)
# X = df.drop(columns=['Screening_date', 'Total_screenings'])
# y = df['Total_screenings']

# print(y.head(100))

# # Se a variável alvo for contínua (regressão)
# mi_scores = mutual_info_regression(X, y)

# # Criar um DataFrame para visualizar os resultados
# mi_df = pd.DataFrame({'Variable': X.columns, 'Mutual Information': mi_scores})
# mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# print(mi_df)

#### results mutual information with Screening_counts per day
#                    Variable  Mutual Information
# 4              Sun_exposure            0.078664
# 9               Phototype_3            0.036020
# 0                    Gender            0.015925
# 11              Phototype_5            0.004084
# 3     Family_cancer_history            0.003707
# 6              Premalignant            0.002653
# 7                 Malignant            0.001945
# 5                Body_signs            0.001458
# 12              Phototype_6            0.001291
# 2   Personal_cancer_history            0.000772
# 10              Phototype_4            0.000253
# 1                       Age            0.000000
# 8               Phototype_2            0.000000


####### TARGET: Screening_count per week

# # Adicionar uma coluna 'Week' que extrai o ano e semana do 'Screening_date'
# df['Week'] = df['Screening_date'].dt.strftime('%Y-%U')

# # Adicionar coluna com o número total de screenings por semana
# df['Total_screenings'] = df.groupby('Week')['Screening_date'].transform('count')

# print(df[['Screening_date', 'Week', 'Total_screenings']])

# print(df)

# # Definir as funções de agregação para cada coluna, mantendo 'Screening_date'
# aggregations = {col: (lambda x: x.mode()[0] if x.dtype == 'O' else x.median()) for col in df.columns if col not in ['Screening_date', 'Week', 'Total_screenings']}
# aggregations['Total_screenings'] = 'first'

# # Agrupar os dados por semana, preservando a coluna 'Screening_date'
# df = df.groupby(['Week', 'Screening_date']).agg(aggregations).reset_index()

# # Exibir o DataFrame resultante
# print(df)

# print(df.columns)

# # Criar uma codificação ordinal para a coluna 'Week'
# df['Week'] = pd.factorize(df['Week'])[0]

# # Definir as features (X) e a variável target (y)
# X = df.drop(columns=['Total_screenings', 'Screening_date', 'Week'])
# y = df['Total_screenings']

# print(X)
# print(y.head(75))

# # Se a variável alvo for contínua (regressão)
# mi_scores = mutual_info_regression(X, y)

# # Criar um DataFrame para visualizar os resultados
# mi_df = pd.DataFrame({'Variable': X.columns, 'Mutual Information': mi_scores})
# mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# print(mi_df)

#### results mutual information with Screening_counts per week
#                    Variable  Mutual Information
# 9               Phototype_3            0.079974
# 4              Sun_exposure            0.054412
# 3     Family_cancer_history            0.015868
# 10              Phototype_4            0.007998
# 7                 Malignant            0.003856
# 12              Phototype_6            0.003184
# 11              Phototype_5            0.002195
# 6              Premalignant            0.001662
# 5                Body_signs            0.001553
# 0                    Gender            0.000000
# 1                       Age            0.000000
# 2   Personal_cancer_history            0.000000
# 8               Phototype_2            0.000000


############### SVM ANOMALIAS/OUTLIERS per day ###################

# from sklearn.svm import OneClassSVM

# # Inicializar o modelo One-Class SVM
# ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)  # nu é uma estimativa da proporção de outliers

# # Treinar o modelo
# ocsvm.fit(X)

# # Prever anomalias
# y_pred = ocsvm.predict(X)

# # y_pred retorna 1 para valores normais e -1 para outliers
# df['Anomaly'] = y_pred

# # Contar o número de anomalias detectadas
# outliers = df[df['Anomaly'] == -1]
# print(f"Anomalias detectadas: {len(outliers)}")

# # Visualizar algumas anomalias (opcional)
# print(outliers.head())

# # Opcional: Visualizar uma distribuição de anomalias
# plt.figure(figsize=(10, 6))
# plt.hist(df['Anomaly'], bins=3, edgecolor='black')
# plt.title('Distribuição de Anomalias (1 = Normal, -1 = Anomalia)')
# plt.xlabel('Anomaly')
# plt.ylabel('Frequency')
# plt.show()

# outliers = df[df['Anomaly'] == -1]
# print(outliers.head(8))

# # Comparar com Dados Normais
# normal_data = df[df['Anomaly'] == 1]
# print(normal_data.describe())
# print(outliers.describe())

# # Ordenar o DataFrame pelo tempo (opcional, mas útil para visualização correta)
# df = df.sort_values(by='Screening_date')

# # Plot daily total screenings
# plt.figure(figsize=(14, 7))
# plt.plot(df['Screening_date'], df['Total_screenings'], label='Total Screenings', color='blue')

# # Highlight the outliers
# outliers = df[df['Anomaly'] == -1]
# plt.scatter(outliers['Screening_date'], outliers['Total_screenings'], color='red', label='Outliers', s=100)

# # Configure the plot
# plt.title('Visualização Temporal dos Screenings por Dia')
# plt.xlabel('Data')
# plt.ylabel('Total de Screenings')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# normal_data = df[df['Anomaly'] == 1]
# print(normal_data.describe())
# print(outliers.describe())


############### SVM ANOMALIAS/OUTLIERS per week ###################

# from sklearn.svm import OneClassSVM

# # Inicializar o modelo One-Class SVM
# ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)  # nu é uma estimativa da proporção de outliers

# # Treinar o modelo
# ocsvm.fit(X)

# # Prever anomalias
# y_pred = ocsvm.predict(X)

# # y_pred retorna 1 para valores normais e -1 para outliers
# df['Anomaly'] = y_pred

# # Contar o número de anomalias detectadas
# outliers = df[df['Anomaly'] == -1]
# print(f"Anomalias detectadas: {len(outliers)}")

# # Visualizar algumas anomalias (opcional)
# print(outliers.head())

# # Opcional: Visualizar uma distribuição de anomalias
# plt.figure(figsize=(10, 6))
# plt.hist(df['Anomaly'], bins=3, edgecolor='black')
# plt.title('Distribuição de Anomalias (1 = Normal, -1 = Anomalia)')
# plt.xlabel('Anomaly')
# plt.ylabel('Frequency')
# plt.show()

# outliers = df[df['Anomaly'] == -1]
# print(outliers.head(8))

# # Comparar com Dados Normais
# normal_data = df[df['Anomaly'] == 1]
# print(normal_data.describe())
# print(outliers.describe())

# # Ordenar o DataFrame pelo tempo (opcional, mas útil para visualização correta)
# df = df.sort_values(by='Week')

# # Gráfico de linhas para todos os dados
# plt.figure(figsize=(14, 7))
# plt.plot(df['Week'], df['Total_screenings'], label='Total Screenings', color='blue')

# # Destacar os outliers no gráfico
# outliers = df[df['Anomaly'] == -1]
# plt.scatter(outliers['Week'], outliers['Total_screenings'], color='red', label='Outliers', s=100)

# # Configurar o gráfico
# plt.title('Visualização Temporal dos Screenings por Semana')
# plt.xlabel('Semana')
# plt.ylabel('Total de Screenings')
# plt.legend()
# plt.grid(True)
# plt.show()


############### STATIONARITY PER DAY ###################

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

# # Estatística do Teste ADF: -2.3969145430811265
# # P-Valor: 0.14260915054092738
# # Número de defasagens usadas no teste: 10
# # Número de observações usadas no teste: 79
# # Valores críticos:
# #         1%: -3.5159766913976376
# #         5%: -2.898885703483903        10%: -2.5866935058484217
# # a série não é estacionária - diferenciação

# # Aplicar diferenciação à série temporal
# df['Total_screenings_diff'] = df['Total_screenings'].diff().dropna()

# # Realizar o teste ADF na série diferenciada
# screenings_diff_series = df['Total_screenings_diff'].dropna()
# result_adf_diff = adfuller(screenings_diff_series)

# print('Estatística do Teste ADF após Diferenciação:', result_adf_diff[0])
# print('P-Valor após Diferenciação:', result_adf_diff[1])

# # Estatística do Teste ADF após Diferenciação: -2.3288037324107114
# # P-Valor após Diferenciação: 0.16284638284498754

# # Aplicar segunda diferenciação à série temporal
# df['Total_screenings_diff2'] = df['Total_screenings_diff'].diff().dropna()

# # Realizar o teste ADF na série diferenciada duas vezes
# screenings_diff2_series = df['Total_screenings_diff2'].dropna()
# result_adf_diff2 = adfuller(screenings_diff2_series)

# print('Estatística do Teste ADF após Segunda Diferenciação:', result_adf_diff2[0])
# print('P-Valor após Segunda Diferenciação:', result_adf_diff2[1])

# # Estatística do Teste ADF após Segunda Diferenciação: -8.427356727606558
# # P-Valor após Segunda Diferenciação: 1.910865522717346e-13

# # Visualização da Série Temporal Estacionária
# plt.figure(figsize=(12, 6))
# plt.plot(df['Total_screenings_diff2'], label='Série Diferenciada Duas Vezes')
# plt.title('Série Temporal Após Segunda Diferenciação')
# plt.xlabel('Data')
# plt.ylabel('Total de Screenings Diferenciado')
# plt.legend()
# plt.show()

############### STATIONARITY PER WEEK ###################

# # Plotar a série temporal dos screenings semanais
# plt.figure(figsize=(14, 7))
# plt.plot(df['Screening_date'], df['Total_screenings'], label='Total Screenings', color='blue')
# plt.title('Screenings Semanais')
# plt.xlabel('Data')
# plt.ylabel('Total de Screenings')
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# from statsmodels.tsa.stattools import adfuller

# # Extraia os valores da série temporal para o teste
# screenings_series = df.set_index('Screening_date')['Total_screenings']

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

# A série é estacionária
# Estatística do Teste ADF: -4.792426282586994
# P-Valor: 5.6229613295990076e-05
# Número de defasagens usadas no teste: 0
# Número de observações usadas no teste: 89
# Valores críticos:
#         1%: -3.506057133647011
#         5%: -2.8946066061911946
#         10%: -2.5844100201994697


############### PEARSON CORRELATION ###################

# # Eliminar a primeira coluna (variável resposta)
# df_sem_resposta = df.iloc[:, 1:]

# # Calcular a correlação de Pearson
# correlacao_pearson = df_sem_resposta.corr(method='pearson')

# print(df.columns)

# # Calcular a correlação de Pearson
# correlacao_pearson = df.corr(method='pearson')

# # Configuração do tamanho das figuras
# plt.figure(figsize=(12, 8))
# # Plotar a matriz de correlação com todas as correlações
# sns.heatmap(correlacao_pearson, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Matriz de Correlação de Pearson - Todas as Variáveis')
# plt.show()
# # Filtrar as correlações fortes (acima de 0.8 ou abaixo de -0.8)
# correlacao_forte = correlacao_pearson[(correlacao_pearson > 0.8) | (correlacao_pearson < -0.8)]
# # Plotar a matriz de correlação com as correlações fortes
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlacao_forte, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Matriz de Correlação de Pearson - Correlações Fortes (>|0.8|)')
# plt.show()









