#Combine excel file with dictionaire
import os
import pandas as pd

# Função para obter todas as colunas de cada planilha de cada arquivo Excel
def get_all_columns(excel_files_dir):
    all_columns = {}  # Dicionário para armazenar colunas de cada arquivo
    
    for filename in os.listdir(excel_files_dir):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(excel_files_dir, filename)
            try:
                # Ler todas as planilhas do arquivo Excel
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    all_columns[file_path + ' - ' + sheet_name] = df.columns.tolist()
            except Exception as e:
                print(f"Erro ao ler o arquivo {filename}: {e}")
    
    return all_columns

# Diretório onde estão localizados os arquivos Excel
excel_files_dir = 'C:\\Users\\Patyc\\OneDrive\\Desktop\\Dissertation\\Data\\Excel Files'

# Obter todas as colunas de cada planilha de cada arquivo Excel
all_columns_info = get_all_columns(excel_files_dir)

# Exibir informações de colunas para cada arquivo e planilha
for file_sheet, columns in all_columns_info.items():
    print(f"Arquivo e planilha: {file_sheet}")
    print(f"Colunas: {columns}")
    print()

#lista de todos os nomes únicos de colunas encontradas
all_unique_columns = set()
for columns in all_columns_info.values():
    all_unique_columns.update(columns)
print("Lista de todos os nomes únicos de colunas encontradas:")
print(all_unique_columns)

import glob

# #Ler e unificar os arquivos Excel
all_files = glob.glob('C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Excel Files')

# # DataFrame para armazenar todos os dados
combined_df = pd.DataFrame()

# for file in all_files:
df = pd.read_excel(all_files)
# Renomear colunas de acordo com o mapeamento
df.rename(columns=column_mapping, inplace=True)
combined_df = pd.concat([combined_df, df], ignore_index=True)

# #Exportar para um único arquivo Excel
combined_df.to_excel("combined_file.xlsx", index=False)
