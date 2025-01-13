import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from typing import List
from time import time
from datetime import timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PatientDataOrganizer:
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the organizer
        
        Args:
            input_dir (str): Directory containing measurement files and patient metadata
            output_dir (str): Directory where the partitioned dataset will be created
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.patient_metadata = None
        
    def read_patient_metadata(self) -> None:
        """Read patient metadata"""
        metadata_path = self.input_dir / 'patient_metadata.csv'
        self.patient_metadata = pd.read_csv(metadata_path)
        logger.info(f"Found {len(self.patient_metadata)} patients in metadata")

    def get_week_id(self, timestamp: pd.Timestamp) -> str:
        """Convert timestamp to week identifier YYYY-WW format"""
        return f"{timestamp.year}-{timestamp.strftime('%V')}"

    def get_measurement_files(self) -> List[Path]:
        """Get all measurement file paths"""
        return sorted(self.input_dir.glob('measurements_part_*.csv'))

    def extract_patient_id(self, device_id: str) -> str:
        """Extract patient ID from device ID"""
        return device_id.split('_')[-1]

    def process_measurements(self) -> None:
        """Process measurements into a partitioned parquet dataset"""
        start_time = time()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create schema from sample data
        sample_df = next(pd.read_csv(self.get_measurement_files()[0], chunksize=1))
        sample_df['patient_id'] = sample_df['device_id'].apply(self.extract_patient_id)
        sample_df['week_id'] = pd.to_datetime(sample_df['timestamp']).apply(self.get_week_id)
        sample_df = sample_df.merge(
            self.patient_metadata,
            on='patient_id',
            how='left'
        )
        
        schema = pa.Schema.from_pandas(sample_df)
        
        # Create partitioning
        partitioning = ds.partitioning(
            pa.schema([
                ('patient_id', pa.string()),
                ('week_id', pa.string())
            ]),
            flavor='hive'
        )
        
        # Create dataset
        dataset = ds.dataset(
            self.output_dir,
            schema=schema,
            partitioning=partitioning
        )
        
        total_rows = 0
        patient_counts = {}
        file_sizes = []
        start_processing = time()
        last_log = start_processing
        
        # Process each measurement file
        for measurement_file in self.get_measurement_files():
            logger.info(f"Processing {measurement_file.name}")
            
            # Read and process in chunks
            for chunk_num, chunk in enumerate(pd.read_csv(measurement_file, chunksize=100000)):
                # Extract patient IDs and merge metadata
                chunk['patient_id'] = chunk['device_id'].apply(self.extract_patient_id)
                
                # Add week partitioning
                chunk['week_id'] = pd.to_datetime(chunk['timestamp']).apply(self.get_week_id)
                
                # Merge metadata
                chunk = chunk.merge(
                    self.patient_metadata,
                    on='patient_id',
                    how='left'
                )
                
                # Update patient counts
                counts = chunk['patient_id'].value_counts()
                for pid, count in counts.items():
                    patient_counts[pid] = patient_counts.get(pid, 0) + count
                
                # Write chunk to dataset
                table = pa.Table.from_pandas(chunk)
                write_options = ds.ParquetFileFormat().make_write_options(
                    compression='snappy',
                    use_dictionary=True,
                    write_statistics=True
                )

                ds.write_dataset(
                    table,
                    self.output_dir,
                    format='parquet',
                    partitioning=partitioning,
                    existing_data_behavior='overwrite_or_ignore',
                    file_options=write_options,
                    max_rows_per_group=500000,  # Control row group size here
                    min_rows_per_group=50000    # Minimum rows per group
                )
                
                total_rows += len(chunk)
                
                # Log progress every 30 seconds
                current_time = time()
                if current_time - last_log >= 30:
                    rate = total_rows / (current_time - start_processing)
                    logger.info(
                        f"Processed {total_rows:,} rows... "
                        f"Rate: {rate:.2f} rows/second"
                    )
                    last_log = current_time
        
        end_time = time()
        elapsed = timedelta(seconds=int(end_time - start_time))
        
        # Calculate and log statistics
        dataset = ds.dataset(self.output_dir, partitioning=partitioning)
        files = list(dataset.files)
        
        # Log dataset statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total processing time: {elapsed}")
        logger.info(f"Total rows processed: {total_rows:,}")
        logger.info(f"Processing rate: {total_rows / (end_time - start_time):.2f} rows/second")
        logger.info(f"Number of partition files: {len(files)}")
        logger.info(f"Number of patients: {len(patient_counts)}")
        
        # Log patient statistics
        patient_stats = pd.Series(patient_counts).describe()
        logger.info("\nMeasurements per Patient:")
        logger.info(f"Mean: {patient_stats['mean']:.2f}")
        logger.info(f"Min: {patient_stats['min']:.0f}")
        logger.info(f"Max: {patient_stats['max']:.0f}")
        
        # Save dataset metadata
        metadata = {
            'total_rows': total_rows,
            'num_patients': len(patient_counts),
            'processing_time': str(elapsed),
            'created_at': pd.Timestamp.now().isoformat(),
            'partitioning': 'patient_id/week_id'
        }
        
        pd.Series(metadata).to_json(self.output_dir / 'dataset_metadata.json')
        
        # Create example queries
        logger.info("\nExample queries:")
        logger.info("""
# Read data for a specific patient and week:
ds.dataset('measurements_dataset').to_table(
    filter=(ds.field('patient_id') == 'P0001') & 
          (ds.field('week_id') == '2025-01')
).to_pandas()

# Read all data for a patient:
ds.dataset('measurements_dataset').to_table(
    filter=(ds.field('patient_id') == 'P0001')
).to_pandas()

# Read all data for a specific week:
ds.dataset('measurements_dataset').to_table(
    filter=(ds.field('week_id') == '2025-01')
).to_pandas()
""")

    def organize_data(self) -> None:
        """Main method to organize all patient data"""
        logger.info("Reading patient metadata...")
        self.read_patient_metadata()
        
        logger.info("Processing measurement files...")
        self.process_measurements()
        
        logger.info("Data organization complete!")

# Example usage
if __name__ == "__main__":
    organizer = PatientDataOrganizer(
        input_dir="patient_dataset",
        output_dir="measurements_dataset"
    )
    organizer.organize_data()