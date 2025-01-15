import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
from typing import Dict, List
import multiprocessing as mp
import json
from kafka.producer import KafkaProducer
from kafka.errors import KafkaError

def create_kafka_producer():
    """
    Create and return a Kafka producer configured with multiple brokers
    """
    return KafkaProducer(
        bootstrap_servers=['localhost:9091', 'localhost:9092', 'localhost:9093'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        retries=5
    )

def process_patient(patient_id: str, dataset_path: str) -> Dict:
    """
    Process data for a single patient using PyArrow dataset and send to Kafka.
    """
    # Create Kafka producer
    producer = create_kafka_producer()
    
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
    
    # Process patient records and send to Kafka
    print(f"\n=== Sending Patient {patient_id} Records to Kafka ===")
    
    topic_name = 'patient_measurements'
    
    for _, row in df.iterrows():
        # Convert row to dictionary and handle datetime serialization
        record = {
            'timestamp': row['timestamp'].isoformat(),
            'measurement_type': row['measurement_type'],
            'raw_value': row['raw_value'],
            'device_id': row['device_id']
        }
        
        # Send to Kafka asynchronously
        future = producer.send(topic_name, value=record)
        try:
            future.get(timeout=10)  # Wait for the send to complete with timeout
        except KafkaError as e:
            print(f"Error sending record to Kafka: {e}")
    
    # Ensure all messages are sent before closing
    producer.flush()
    producer.close()
    
    results = {
        'patient_id': patient_id,
        'total_records': len(df),
        'first_adjusted_timestamp': df['adjusted_timestamp'].min().isoformat(),
        'last_adjusted_timestamp': df['adjusted_timestamp'].max().isoformat(),
        'max_delay': float(df['delay'].max()),
        'avg_delay': float(df['delay'].mean())
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
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()