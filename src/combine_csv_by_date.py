import os
import pandas as pd
from datetime import datetime

# Directory containing the CSV files
data_dir = './data/'

# Get today's date
today = datetime.today().date()

# Get list of all CSV files in the directory
csv_files = [f for f in os.listdir(data_dir) if f.startswith('conversation_log') and f.endswith('.csv')]

# List to hold dataframes for today's date
today_dfs = []

# Process each file
for csv_file in csv_files:
    
    # Extract date from the filename (assuming the date is in the format 'YYYYMMDD' at the start)
    date_str = csv_file.split('_')[2].split('.')[0]  # Adjusted split to handle file extension
    file_date = datetime.strptime(date_str, '%Y%m%d').date()
    
    # Check if the file's date is today
    if file_date == today:
        # Read the CSV file
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path)
        today_dfs.append(df)

# Combine all dataframes for today's date
if today_dfs:
    combined_df = pd.concat(today_dfs)
    output_file = os.path.join(data_dir, f'combined_log_{today}.csv')
    combined_df.to_csv(output_file, index=False)
    print(f'Saved combined file for {today} to {output_file}')
else:
    print(f'No CSV files found for today ({today}).')
