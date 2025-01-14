import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from typing import List
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
        self.parquet_output_dir = Path(f'{output_dir}/patients')
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
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_output_dir.mkdir(parents=True, exist_ok=True)

        # Read all measurement files
        measurement_files = self.get_measurement_files()
        logger.info(f"Reading {len(measurement_files)} measurement files...")
        
        # Read and concatenate all files
        dfs = []
        for file in measurement_files:
            logger.info(f"Reading {file.name}")
            df = pd.read_csv(file)
            dfs.append(df)
        
        # Combine all data
        logger.info("Combining all data...")
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total rows: {len(combined_df):,}")

        # Process the data
        logger.info("Processing data...")
        combined_df['patient_id'] = combined_df['device_id'].apply(self.extract_patient_id)
        combined_df['week_id'] = pd.to_datetime(combined_df['timestamp']).apply(self.get_week_id)
        
        # Merge with metadata
        logger.info("Merging with patient metadata...")
        combined_df = combined_df.merge(
            self.patient_metadata,
            on='patient_id',
            how='left'
        )

        # Create partitioning
        partitioning = ds.partitioning(
            pa.schema([
                ('patient_id', pa.string()),
                ('week_id', pa.string())
            ]),
            flavor='hive'
        )

        # Convert to table
        logger.info("Converting to Arrow table...")
        table = pa.Table.from_pandas(combined_df)

        # Write options
        write_options = ds.ParquetFileFormat().make_write_options(
            compression='snappy',
            use_dictionary=True,
            write_statistics=True
        )

        # Write dataset
        logger.info("Writing partitioned dataset...")
        ds.write_dataset(
            table,
            self.parquet_output_dir,
            format='parquet',
            partitioning=partitioning,
            existing_data_behavior='delete_matching',
            file_options=write_options
        )

        # Log statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total rows processed: {len(combined_df):,}")
        logger.info(f"Number of patients: {combined_df['patient_id'].nunique():,}")
        logger.info(f"Number of weeks: {combined_df['week_id'].nunique():,}")
        
        # Save dataset metadata
        metadata = {
            'total_rows': len(combined_df),
            'num_patients': combined_df['patient_id'].nunique(),
            'num_weeks': combined_df['week_id'].nunique(),
            'created_at': pd.Timestamp.now().isoformat(),
            'partitioning': 'patient_id/week_id'
        }
        
        pd.Series(metadata).to_json(self.output_dir / 'dataset_metadata.json')

    def organize_data(self) -> None:
        """Main method to organize all patient data"""
        logger.info("Starting data organization process...")
        
        try:
            logger.info("Reading patient metadata...")
            self.read_patient_metadata()
            
            logger.info("Processing measurement files...")
            self.process_measurements()
            
            logger.info("Data organization completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during data organization: {e}")
            raise

if __name__ == "__main__":
    organizer = PatientDataOrganizer(
        input_dir="patient_dataset",
        output_dir="measurements_dataset"
    )
    organizer.organize_data()