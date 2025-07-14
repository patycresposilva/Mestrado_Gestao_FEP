import os
import pandas as pd

def remove_columns_from_excel_files(directory, columns_to_remove):
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(directory, filename)
            
            # Load the Excel file
            try:
                df = pd.read_excel(file_path)
                # Remove specified columns if they exist
                df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
                
                # Save the updated DataFrame back to the Excel file
                df.to_excel(file_path, index=False)
                print(f"Processed file: {filename}")
            except Exception as e:
                print(f"Failed to process file: {filename}. Error: {e}")

directory = r'C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Excel Files'

columns_to_remove = ['Hora marcação', 'TO', 'Como soube deste/s rastreio/s?', 'Como soube deste rastreio?']

remove_columns_from_excel_files(directory, columns_to_remove)