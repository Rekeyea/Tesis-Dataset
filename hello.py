import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

def read_latest_measurements(base_path: str, patient_id: str, limit: int = 10):
    """
    Read latest measurements for a patient
    
    Args:
        base_path (str): Path to the dataset directory
        patient_id (str): Patient ID to query
        limit (int): Number of measurements to return
    """
    dataset = ds.dataset(
        source=base_path,
        format="parquet",
        partitioning=ds.HivePartitioning(
            schema=pa.schema([
                ("patient_id", pa.string()),
                ("week_id", pa.string())
            ])
        ),
        exclude_invalid_files=True
    )
    
    # Read data and convert to pandas
    df = dataset.to_table(
        filter=ds.field('patient_id') == patient_id
    ).to_pandas()
    
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp descending and get top N
    return df.sort_values('timestamp', ascending=False).head(limit)

# Get latest 10 measurements
df = read_latest_measurements("measurements_dataset", "P0001")
print("\nLatest measurements for P0001:")
print(df[['timestamp', 'measurement_type', 'raw_value', 'device_id']])