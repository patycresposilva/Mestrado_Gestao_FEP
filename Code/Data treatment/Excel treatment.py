#data treatment

import pandas as pd
import numpy as np
import os

df = pd.read_excel('C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Merged_File_v7_skin_clean.xlsx')

# Adicionar a nova coluna binária para Skin_diagnosis
df['Premalignant'] = df['Skin_diagnosis'].apply(lambda x: 1 if x == 2 else 0)
df['Malignant'] = df['Skin_diagnosis'].apply(lambda x: 1 if x==3 else 0)

# Adicionar a nova coluna binária para Phototype
df['Phototype_2'] = df['Phototype'].apply(lambda x: 1 if x == 2 else 0)
df['Phototype_3'] = df['Phototype'].apply(lambda x: 1 if x == 3 else 0)
df['Phototype_4'] = df['Phototype'].apply(lambda x: 1 if x == 4 else 0)
df['Phototype_5'] = df['Phototype'].apply(lambda x: 1 if x == 5 else 0)
df['Phototype_6'] = df['Phototype'].apply(lambda x: 1 if x == 6 else 0)

# Transformar a coluna binárias de 'y'/'n' para 1/0
df['Gender'] = df['Gender'].map({'f': 1, 'm': 0})
df['Personal_cancer_history'] = df['Personal_cancer_history'].map({'y': 1, 'n': 0})
df['Family_cancer_history'] = df['Family_cancer_history'].map({'y': 1, 'n': 0})
df['Sun_exposure'] = df['Sun_exposure'].map({'y': 1, 'n': 0})
df['Body_signs'] = df['Body_signs'].map({'y': 1, 'n': 0})

# Remover colunas 'Phototype' e 'Skin_diagnosis'
df = df.drop(columns=['Phototype', 'Skin_diagnosis'])

# print(df.head(10))

# Definir o caminho para salvar o arquivo Excel
output_folder = r"C:\\Users\\Patyc\\OneDrive\\Desktop\\Dissertation\\Data\\Skin_clean and treated"
if not os.path.exists(output_folder): os.makedirs(output_folder)
output_file = os.path.join(output_folder, "Skin_clean.xlsx")

# Salvar o DataFrame em um arquivo Excel
df.to_excel(output_file, index=False)





