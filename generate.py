from enum import Enum
import pandas as pd
import numpy as np
import os
from typing import List
import json

from generation.devices import AVAILABLE_DEVICES, Device, DeviceType
from generation.patients import PATIENT_TEMPLATES, Patient, PatientTemplate, PatientTemplateType
from generation.vitals import ConsciousnessRange

class CareLevel(str, Enum):
    SELF_MONITORING = "SELF_MONITORING"
    HOME_MONITORING = "HOME_MONITORING"
    INTERMEDIATE_CARE = "INTERMEDIATE_CARE"
    INTENSIVE_CARE = "INTENSIVE_CARE"

# Define care level device sets
CARE_LEVEL_DEVICE_SETS = {
    CareLevel.SELF_MONITORING: {
        DeviceType.SMARTWATCH_BASIC,
        DeviceType.BP_CUFF_AUTO
    },
    CareLevel.HOME_MONITORING: {
        DeviceType.SMARTWATCH_PREMIUM,
        DeviceType.BP_CUFF_AUTO,
        DeviceType.MANUAL_NURSE
    },
    CareLevel.INTERMEDIATE_CARE: {
        DeviceType.SMARTWATCH_PREMIUM,
        DeviceType.PORTABLE_MONITOR,
        DeviceType.CHEST_PATCH,
        DeviceType.TEMP_PATCH,
        DeviceType.MANUAL_NURSE
    },
    CareLevel.INTENSIVE_CARE: {
        DeviceType.MULTIMONITOR_PRO,
        DeviceType.CHEST_PATCH,
        DeviceType.TEMP_PATCH,
        DeviceType.MANUAL_NURSE
    }
}

class MeasurementGenerator:
    def __init__(self, patient: Patient):
        self.patient = patient
        
    def generate_dataset(self, start_time: pd.Timestamp, 
                        end_time: pd.Timestamp) -> pd.DataFrame:
        """Generate complete dataset for all devices"""
        all_measurements = []
        
        for device in self.patient.assigned_devices:
            device_measurements = []
            
            # Generate measurements for each supported vital sign
            for vital_sign in device.supported_measurements:
                # Generate timestamps for this vital sign
                timestamps = device.generate_timestamps(vital_sign, start_time, end_time)
                n_measurements = len(timestamps)
                
                # Generate measurement values
                vital_range = self.patient.template.vital_ranges[vital_sign]
                if isinstance(vital_range, ConsciousnessRange):
                    values = vital_range.generate_values(
                        n_measurements, 
                        deterioration_probability=self.patient.template.deterioration_probability
                    )
                else:
                    values = vital_range.generate_values(
                        n_measurements, 
                        deterioration_probability=self.patient.template.deterioration_probability
                    )
                
                # Create measurements DataFrame
                measurements = pd.DataFrame({
                    'device_id': f"{device.device_id}_{self.patient.patient_id}",
                    'measurement_type': vital_sign,
                    'timestamp': timestamps,
                    'raw_value': values,
                    'battery': self._generate_battery_levels(device, timestamps),
                    'signal_strength': self._generate_signal_strength(device, n_measurements),
                    'delay': self._generate_delays(device, n_measurements)
                })
                
                device_measurements.append(measurements)
            
            if device_measurements:
                all_measurements.append(pd.concat(device_measurements))
        
        return pd.concat(all_measurements).sort_values('timestamp')

    def _generate_battery_levels(self, device: Device, 
                               timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Generate battery levels with realistic drain"""
        duration_hours = np.diff(timestamps) / np.timedelta64(1, 'h')
        drain = np.concatenate([[0], duration_hours * device.battery_drain_rate])
        battery_levels = 100 - np.cumsum(drain)
        return np.maximum(0, battery_levels)

    def _generate_signal_strength(self, device: Device, n: int) -> np.ndarray:
        """Generate signal strength with random walk variation"""
        base_signal = 0.9
        variations = np.random.uniform(-device.signal_variance, 
                                     device.signal_variance, n)
        signal = np.clip(base_signal + np.cumsum(variations), 0, 1)
        return signal * 0.3 + base_signal * 0.7

    def _generate_delays(self, device: Device, n: int) -> np.ndarray:
        """Generate delays based on probability"""
        delays = np.zeros(n)
        mask = np.random.random(n) < device.delay_probability
        delays[mask] = np.random.randint(1, device.max_delay + 1, mask.sum())
        return delays

def generate_patient_population(n_patients: int = 100) -> List[Patient]:
    """
    Generate a population of patients based on predefined probabilities
    for patient templates and care levels.
    
    Args:
        n_patients: Number of patients to generate
    Returns:
        List of Patient objects with assigned templates and devices
    """
    # Define probability distribution for patient templates
    template_probabilities = {
        PatientTemplateType.HEALTHY: 0.5,           # Most patients are healthy
        PatientTemplateType.STABLE_CHRONIC: 0.3,    # Significant number with chronic conditions
        PatientTemplateType.DETERIORATING: 0.2,     # Some with worsening conditions
    }
    
    # Define care level probabilities for each template
    care_level_mapping = {
        PatientTemplateType.HEALTHY: {
            CareLevel.SELF_MONITORING: 0.7,
            CareLevel.HOME_MONITORING: 0.3,
            CareLevel.INTERMEDIATE_CARE: 0.0,
            CareLevel.INTENSIVE_CARE: 0.0
        },
        PatientTemplateType.STABLE_CHRONIC: {
            CareLevel.SELF_MONITORING: 0.5,
            CareLevel.HOME_MONITORING: 0.5,
            CareLevel.INTERMEDIATE_CARE: 0,
            CareLevel.INTENSIVE_CARE: 0
        },
        PatientTemplateType.DETERIORATING: {
            CareLevel.SELF_MONITORING: 0.2,
            CareLevel.HOME_MONITORING: 0.3,
            CareLevel.INTERMEDIATE_CARE: 0.3,
            CareLevel.INTENSIVE_CARE: 0.2
        }
    }
    
    patients = []
    
    # Validate probability distributions
    if not np.isclose(sum(template_probabilities.values()), 1.0, rtol=1e-5):
        raise ValueError("Template probabilities must sum to 1")
    
    for care_level_probs in care_level_mapping.values():
        if not np.isclose(sum(care_level_probs.values()), 1.0, rtol=1e-5):
            raise ValueError("Care level probabilities must sum to 1 for each template")
    
    # Generate patients
    for i in range(n_patients):
        patient_id = f"P{str(i+1).zfill(4)}"  # P0001, P0002, etc.
        
        # Select template based on probabilities
        template_id = np.random.choice(
            list([k.name for k in template_probabilities.keys()]),
            p=list(template_probabilities.values())
        )
        
        # Select care level based on template-specific probabilities
        care_level = np.random.choice(
            list([k.name for k in care_level_mapping[template_id].keys()]),
            p=list(care_level_mapping[template_id].values())
        )
        
        # Create patient with selected template and care level
        template = PATIENT_TEMPLATES[template_id]
        devices = [AVAILABLE_DEVICES[dev_id.name] for dev_id in CARE_LEVEL_DEVICE_SETS[care_level]]
        
        patient = Patient(
            patient_id=patient_id,
            template=template,
            assigned_devices=devices
        )
        
        patients.append(patient)
    
    return patients

def generate_complete_dataset(
    n_patients: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    output_dir: str = "dataset"
) -> None:
    """
    Generate and save a complete dataset with both simplified patient metadata
    and detailed dataset information.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating data for {n_patients} patients from {start_time} to {end_time}")
    
    # Generate patient population
    print("Generating patient population...")
    patients = generate_patient_population(n_patients)
    
    # Generate measurements and collect metadata
    print("Generating measurements...")
    all_measurements = []
    patient_metadata = []
    total_measurements = 0
    
    for i, patient in enumerate(patients, 1):
        print(f"Processing patient {i}/{n_patients}: {patient.patient_id}")
        
        # Generate measurements
        generator = MeasurementGenerator(patient)
        measurements = generator.generate_dataset(start_time, end_time)
        all_measurements.append(measurements)
        total_measurements += len(measurements)
        
        # Collect simplified metadata
        devices = patient.assigned_devices
        device_ids = {dev.device_id for dev in devices}
        care_level = next(level for level, devs in CARE_LEVEL_DEVICE_SETS.items() 
                         if device_ids == devs)
        
        metadata = {
            'patient_id': patient.patient_id,
            'template_id': patient.template.template_id,
            'care_level': care_level
        }
        patient_metadata.append(metadata)
        
        # Save partial measurements every 10 patients
        if i % 10 == 0:
            partial_df = pd.concat(all_measurements, ignore_index=True)
            partial_df.to_csv(os.path.join(output_dir, f'measurements_part_{i//10}.csv'), 
                            index=False, float_format='%.3f')
            all_measurements = []  # Clear memory
            print(f"Saved part {i//10} with {len(partial_df)} measurements")
    
    # Save any remaining measurements
    if all_measurements:
        partial_df = pd.concat(all_measurements, ignore_index=True)
        partial_df.to_csv(os.path.join(output_dir, f'measurements_part_{(n_patients//10)+1}.csv'), 
                         index=False, float_format='%.3f')
    
    # Save simplified metadata
    metadata_df = pd.DataFrame(patient_metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'patient_metadata.csv'), index=False)
    
    # Save detailed dataset info
    dataset_info = {
        'n_patients': n_patients,
        'total_measurements': total_measurements,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'measurement_duration_hours': (end_time - start_time).total_seconds() / 3600,
        'avg_measurements_per_patient': total_measurements / n_patients,
        'template_distribution': metadata_df['template_id'].value_counts().to_dict(),
        'care_level_distribution': metadata_df['care_level'].value_counts().to_dict(),
        'device_distribution': {
            device_id: len([p for p in patients if any(d.device_id == device_id for d in p.assigned_devices)])
            for device_id in set(d.device_id for p in patients for d in p.assigned_devices)
        }
    }
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\nDataset generation complete!")
    print(f"Total measurements: {total_measurements:,}")
    print(f"Average measurements per patient: {total_measurements/n_patients:,.1f}")
    print(f"\nFiles saved in directory: {output_dir}/")
    print("- measurements_part_*.csv (measurement data split into parts)")
    print("- patient_metadata.csv (simplified patient information)")
    print("- dataset_info.json (detailed dataset statistics)")

if __name__ == "__main__":
    # Generate one week of data
    start_time = pd.Timestamp('2025-01-08 00:00:00')
    end_time = start_time + pd.Timedelta(days=7)
    
    # Generate dataset with 100 patients
    generate_complete_dataset(
        n_patients=10,
        start_time=start_time,
        end_time=end_time,
        output_dir="patient_dataset"
    )