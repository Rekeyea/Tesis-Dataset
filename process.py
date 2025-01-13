import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientDataProcessor:
    def __init__(self, base_path: str):
        """
        Initialize the processor with the dataset path
        
        Args:
            base_path (str): Path to the dataset directory
        """
        self.base_path = base_path
        self.dataset = ds.dataset(
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

    def normalize_patient_id(self, patient_id: str) -> str:
        """
        Normalize patient ID format
        
        Args:
            patient_id (str): Patient ID to normalize
            
        Returns:
            str: Normalized patient ID
        """
        # Remove any extra zeros between P and the number
        if patient_id.startswith('P00'):
            number = int(patient_id[3:])
            return f"P{number:04d}"
        return patient_id

    def read_patient_measurements(self, patient_id: str) -> pd.DataFrame:
        """
        Read all measurements for a single patient
        
        Args:
            patient_id (str): Patient ID to query
            
        Returns:
            pd.DataFrame: DataFrame containing patient measurements
        """
        try:
            normalized_id = self.normalize_patient_id(patient_id)
            
            # First try with normalized ID
            df = self.dataset.to_table(
                filter=ds.field('patient_id') == normalized_id
            ).to_pandas()
            
            if len(df) == 0 and normalized_id != patient_id:
                # If no data found, try with original ID
                df = self.dataset.to_table(
                    filter=ds.field('patient_id') == patient_id
                ).to_pandas()
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            logger.info(f"Successfully read data for patient {patient_id} (normalized to {normalized_id}): {len(df)} records")
            
            # Verify data integrity
            if len(df) == 0:
                logger.warning(f"No data found for patient {patient_id} (normalized to {normalized_id})")
            elif len(df) < 14000:  # Assuming this is significantly lower than expected
                logger.warning(f"Unusually low number of records for patient {patient_id}: {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {str(e)}")
            raise

    def process_patients_parallel(self, patient_ids: List[str], max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Process multiple patients in parallel
        
        Args:
            patient_ids (List[str]): List of patient IDs to process
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping patient IDs to their processed data
        """
        results = {}
        
        # First, normalize all patient IDs
        normalized_ids = [(pid, self.normalize_patient_id(pid)) for pid in patient_ids]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with normalized IDs
            future_to_patient = {
                executor.submit(self.read_patient_measurements, norm_id): orig_id 
                for orig_id, norm_id in normalized_ids
            }
            
            # Process completed tasks in order of completion
            for future in future_to_patient:
                patient_id = future_to_patient[future]
                try:
                    results[patient_id] = future.result()
                except Exception as e:
                    logger.error(f"Failed to process patient {patient_id}: {str(e)}")
                    results[patient_id] = None
        
        return results

    def get_combined_timeline(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all patient data into a single timeline
        
        Args:
            processed_data (Dict[str, pd.DataFrame]): Dictionary of processed patient data
            
        Returns:
            pd.DataFrame: Combined timeline of all measurements
        """
        # Combine all valid dataframes
        valid_dfs = [df for df in processed_data.values() if df is not None and len(df) > 0]
        if not valid_dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        timeline = combined_df.sort_values('timestamp')
        
        # Verify total number of records
        total_records = len(timeline)
        logger.info(f"Combined timeline contains {total_records} total records")
        
        return timeline

# Example usage
def main():
    processor = PatientDataProcessor("measurements_dataset")
    
    # List of patient IDs to process (using correct format)
    patient_ids = [f"P{i:04d}" for i in range(1, 11)]  # P0001 to P0010
    
    # Process all patients in parallel
    processed_data = processor.process_patients_parallel(patient_ids)
    
    # Get combined timeline
    timeline = processor.get_combined_timeline(processed_data)
    
    # Print summary
    print("\nCombined timeline summary:")
    print(f"Total measurements: {len(timeline)}")
    print("\nFirst 10 measurements across all patients:")
    print(timeline[['timestamp', 'patient_id', 'measurement_type', 'raw_value', 'device_id']].head(10))
    
    # Print individual patient summaries
    print("\nPer-patient summary:")
    for patient_id, df in processed_data.items():
        if df is not None:
            print(f"{patient_id}: {len(df)} measurements")
        else:
            print(f"{patient_id}: Processing failed")
    
    # Print total records check
    print(f"\nTotal records check: {sum(len(df) for df in processed_data.values() if df is not None)}")

if __name__ == "__main__":
    main()