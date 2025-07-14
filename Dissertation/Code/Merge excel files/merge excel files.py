# merge excel files

import os
import pandas as pd

def merge_excel_files(directory, output_file):
    # List to hold all DataFrames
    all_data = []
    
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            file_path = os.path.join(directory, filename)
            
            # Load the Excel file
            try:
                df = pd.read_excel(file_path)
                # Add a new column for the filename
                df['Filename'] = filename
                all_data.append(df)
                print(f"Loaded file: {filename}")
            except Exception as e:
                print(f"Failed to load file: {filename}. Error: {e}")
    
    # Concatenate all DataFrames
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # Save the merged DataFrame to a new Excel file
        try:
            merged_df.to_excel(output_file, index=False)
            print(f"Merged file saved as: {output_file}")
        except Exception as e:
            print(f"Failed to save merged file. Error: {e}")
    else:
        print("No Excel files found in the directory.")

# Example usage
directory = r'C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Excel Files'
output_file = r'C:/Users/Patyc/OneDrive/Desktop/Dissertation/Data/Merged_File.xlsx'

merge_excel_files(directory, output_file)
