import pandas as pd
from pathlib import Path

def concatenate_csv(input_folder):
    # Define the path to the input data and output directory
    raw_data_path = Path(input_folder)
    processed_data_path = Path('data/processed')

    # Create the output directory if it doesn't exist
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # Read all CSV files in the input data directory
    all_files = raw_data_path.glob('*.csv')

    # Concatenate all DataFrames
    combined_df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
        # Group by 'task_id' and save each group as a separate CSV
    for task_id, group in combined_df.groupby('TaskPrompt'):
        group.to_csv(processed_data_path / f'task_{task_id}.csv', index=False)
        

def save_response_text_only(concatenated_file_path):
    # Check if the file exists
    if not Path(concatenated_file_path).exists():
        print(f"File not found: {concatenated_file_path}")
        return
    
    concatenated_df = pd.read_csv(concatenated_file_path)
    
    # Check if ResponseText column exists
    if 'ResponseText' not in concatenated_df.columns:
        print("ResponseText column is not found in the concatenated CSV.")
        return

    # Extract the ResponseText column
    response_text_df = concatenated_df[['ResponseText']]

    # Define the output path for the SLURM folder
    output_path = Path('slurm/response_text.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save without header
    response_text_df.to_csv(output_path, index=False, header=False)

    print(f"Response text saved to {output_path}")