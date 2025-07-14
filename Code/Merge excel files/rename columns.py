#rename columns

import os
import pandas as pd

def rename_columns_in_excel_files(directory, rename_rules):
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            file_path = os.path.join(directory, filename)

            # Load the Excel file
            try:
                df = pd.read_excel(file_path)
                
                # Rename columns according to the provided rules
                df.rename(columns=rename_rules, inplace=True)
                
                # Save the updated DataFrame back to the Excel file
                df.to_excel(file_path, index=False)
                print(f"Processed file: {filename}")
            except Exception as e:
                print(f"Failed to process file: {filename}. Error: {e}")

# Example usage
directory = r'C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Excel Files'

rename_rules = {
    ('rastreio', 'Inscrito nos 2 Rastreios?', 'RASTREIOS', 'rastreios', 'Qual o rastreio?', 'Qual o rastreio?'): 'rastreio',
    ('Sexo', 'Género'): 'Género',
    ('Data de nascimento', 'Data Nascimeto', 'Data Nascimento', 'Data Nascimento'): 'Data de nascimento',
    'Profissão': 'Profissão',
    'Habilitações Literárias': 'Educação',
    ('Frequência de exposição solar?', 'Frequência de exposição solar?', 'Frquência de exposição solar?', 'Profissão tem ou teve grande exposição solar?'): 'Profissão tem ou teve grande exposição solar?',
    ('histórico pessoal de cancro', 'História pessoal de cancro?'): 'Histórico pessoal de cancro',
    ('se sim, qual?', 'se sim, qual?', 'Se sim, qual?', 'Se sim, qual?'): 'Se sim, qual?',
    ('História cancro de pele na família?', 'Histórial cancro de pele na familia?', 'Historial de cancro na família?', 'Historial cancro de pele na familia? Quem?'): 'Historial de cancro de pele na família?',
    'Quem?':'Quem?',
    ('Mais de 50 sinais?', 'Mais de 50 sinais?'): 'Mais de 50 sinais',
    ('Fotótipo (ver tabela anexo) ', 'Fotótipo (ver tabela em anexo)', 'Fotótipo (ver tabela anexo)' , 'Fotótipo (ver tabela em anexo)'): 'Fotótipo',
    'Diagnóstico RP': 'Diagnóstico RP',
    ('Observação RP', 'Observação'): 'Observação RP',
    ('Fumador/a?', 'É fumador?'): 'Fumador/a?',
    ('É fumador ou já foi?', 'fuma? Ou já fumou?', 'fuma? Ou já fumou?', 'Fuma ou fumou no passado?'): 'Fuma ou já fumou?',
    ('Já foi no passado?', 'Fumador/a passado?', 'JÁ FOI?'): 'Fumador/a passado?',
    ('Se sim, quantos?', 'Se sim, quantos cigarros por dia?', 'SSe sim, quantos cigarros p/ dia?'): 'Se sim, quantos cigarros',
    ('bebidas alcoólicas?', 'Consome bebidas alcoólica?', 'Consome bebidas alcoólicas?' , 'Consumo alcool?', 'bebidas alcoólicas?', 'Bebidas alcoolicas?', 'Bebidas alcoolicas?'): 'Consome bebidas alcoólicas',
    ('se sim, com que frequência?',  'com que frequência?.1', 'Se sim, com que frequência?', 'com que frequência?', 'Se sim, frequência?',   'Com que frequência?'): 'Se sim, com que frequência?',
    ('Diagnóstico RO', 'diagnóstico RO'): 'Diagnóstico RO',
    ('Observação RO', 'Observação.1', 'Observações RO'): 'Observação RO'
}
rename_columns_in_excel_files(directory, rename_rules)
