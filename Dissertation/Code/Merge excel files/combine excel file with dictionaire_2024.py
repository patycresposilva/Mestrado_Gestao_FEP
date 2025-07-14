#combine excel file with dictionaire_2024

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
excel_files_dir = 'C:\\Users\\Patyc\\OneDrive\\Desktop\\Dissertation\\Data\\Excel Files\\2024'

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

def unify_excels_simple(directory_path, output_file):
    all_data = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_excel(file_path)
            df['Source_File'] = filename
            all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_excel(output_file, index=False)
    print(f"All Excel files have been unified into {output_file}")

# Example usage:
unify_excels_simple('C:\\Users\\Patyc\\OneDrive\\Desktop\\Dissertation\\Data\\Excel Files\\2024', 'combined_excel_2024.xlsx')

