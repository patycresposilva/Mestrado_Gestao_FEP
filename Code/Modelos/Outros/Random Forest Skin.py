#Random_Forest_Skin

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib

# Carregar os dados
data = pd.read_excel('C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Merged_File_v7_skin.xlsx')

# Definir o alvo e as características
y = data['Screening_date']
X = data.drop(columns=['Location', 'Screening_type', 'Birth date', 'Profession', 'Education', 'If so, which one?', 'If so, who?', 'If so, which one?2', 'Skin_observations', 'Screening_date'])

# Converter Screening_date para o número de dias desde uma data de referência
reference_date = pd.Timestamp('2022-01-01')
y = (data['Screening_date'] - reference_date).dt.days

# Verificar se o número de linhas é consistente
print("Número de linhas em features:", len(X))
print("Número de linhas no y:", len(y))

# Verificar valores únicos nas colunas categóricas antes da conversão
categorical_features = ['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Sun_exposure', 'Body_signs', 'Phototype', 'Skin_diagnosis']
for col in categorical_features:
    unique_values = data[col].unique()
    print(f"Valores únicos na coluna {col} antes da conversão: {unique_values}")

# Ajustar os tipos de dados
data['Screening_date'] = pd.to_datetime(data['Screening_date'], errors='coerce')
data['Gender'] = data['Gender'].astype('str')
data['Personal_cancer_history'] = data['Personal_cancer_history'].astype('str')
data['Family_cancer_history'] = data['Family_cancer_history'].astype('str')
data['Sun_exposure'] = data['Sun_exposure'].astype('str')
data['Body_signs'] = data['Body_signs'].astype('str')
data['Phototype'] = data['Phototype'].astype('str')
data['Skin_diagnosis'] = data['Skin_diagnosis'].astype('str')

# Handle null values
# Definindo as colunas numéricas e categóricas
numeric_features = ['Age']
categorical_features = ['Gender', 'Personal_cancer_history', 'Family_cancer_history', 'Sun_exposure', 'Body_signs', 'Phototype', 'Skin_diagnosis']

# Preprocessor para tratar os dados
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first'))
        ]), categorical_features)
    ],
    remainder='passthrough'  # Passar as colunas que não foram especificadas
)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline de pré-processamento e modelagem
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Treinar o modelo
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)
print("Avaliação do modelo:")
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R^2: {r2_score(y_test, y_pred)}')

# Opcional: Converter previsões de volta para datas
y_pred_dates = pd.to_datetime(y_pred, origin=reference_date, unit='D')
print("Previsões de datas:", y_pred_dates)

# Salvar o modelo
joblib.dump(model, 'skin_cancer_screening_model1_RF.joblib')

