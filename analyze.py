import pandas as pd
import glob
import os

def count_devices_ending_with(folder_path, ending_string):
    try:
        # Create pattern for matching CSV files
        file_pattern = os.path.join(folder_path, 'measurements_part_*.csv')
        
        # Get list of all matching files
        csv_files = glob.glob(file_pattern)
        
        if not csv_files:
            print(f"No CSV files found matching pattern in {folder_path}")
            return 0
            
        total_matching_rows = 0
        
        # Read each CSV file
        for file in csv_files:
            # Read CSV with specified columns
            df = pd.read_csv(file, 
                           names=['device_id', 'measurement_type', 'timestamp', 
                                 'raw_value', 'battery', 'signal_strength', 'delay'])
            
            # Count rows where device_id ends with the specified string
            matching_rows = df[df['device_id'].astype(str).str.endswith(ending_string)].shape[0]
            total_matching_rows += matching_rows
            
            print(f"Found {matching_rows:,} matching devices in {os.path.basename(file)}")
            
        print(f"\nTotal rows with device IDs ending in '{ending_string}': {total_matching_rows:,}")
        return total_matching_rows
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    folder_path = 'patient_dataset'
    ending_string = '_P0008'  # Replace with your desired ending string
    count_devices_ending_with(folder_path, ending_string)