import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
from typing import Dict, List
import multiprocessing as mp

def process_patient(patient_id: str, dataset_path: str) -> Dict:
    """
    Process data for a single patient using PyArrow dataset.
    """
    dataset = ds.dataset(dataset_path, format='parquet')
    patient_filter = pc.match_substring_regex(ds.field('device_id'), f".*_{patient_id}$")
    
    # Read only this patient's data
    df = dataset.to_table(filter=patient_filter).to_pandas()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert delay to float using numpy
    import numpy as np
    delays = np.array(df['delay'].astype(str).str.replace('.000', ''), dtype=float)
    
    # Create adjusted timestamp using numpy timedelta64
    df['adjusted_timestamp'] = df['timestamp'] + np.array(delays * 1e9, dtype='timedelta64[ns]')
    
    # Store the float delays
    df['delay'] = delays
    
    # Sort by adjusted timestamp
    df = df.sort_values('adjusted_timestamp')
    
    # Print patient records in order
    print(f"\n=== Patient {patient_id} Records (Ordered by Timestamp + Delay) ===")
    for _, row in df.iterrows():
        # Here we can Print or send to Kafka
        print(f"Adjusted Timestamp: {row['adjusted_timestamp']} Timestamp: {row['timestamp']}, Measurement: {row['measurement_type']}, Reading: {row['raw_value']}, Device: {row['device_id']}", end='\r\n')
        
    results = {
        'patient_id': patient_id,
        'total_records': len(df),
        'first_adjusted_timestamp': df['adjusted_timestamp'].min(),
        'last_adjusted_timestamp': df['adjusted_timestamp'].max(),
        'max_delay': df['delay'].max(),
        'avg_delay': df['delay'].mean()
    }
    return results

def get_partition_values(dataset_path: str) -> list:
    dataset = ds.dataset(dataset_path, format='parquet')
    scanner = dataset.scanner(columns=['device_id'])
    table = scanner.to_table()
    device_ids = table['device_id'].to_pylist()
    patient_ids = sorted(set(device_id.split('_')[-1] for device_id in device_ids))
    return patient_ids

def process_all_patients(dataset_path: str) -> List[Dict]:
    """
    Process all patients in parallel.
    """
    patient_ids = get_partition_values(dataset_path)
    print(f"Found {len(patient_ids)} unique patient IDs")
    
    # Create a pool of workers
    with mp.Pool(processes=len(patient_ids)) as pool:
        tasks = [(patient_id, dataset_path) for patient_id in patient_ids]
        results = pool.starmap(process_patient, tasks)
    
    return results

def main():
    dataset_path = "measurements_dataset/patients"
    results = process_all_patients(dataset_path)
    print(results)

if __name__ == "__main__":
    main()