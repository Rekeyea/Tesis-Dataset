from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set
from generation.vitals import VitalSign

import numpy as np
import pandas as pd



@dataclass
class Device:
    """Defines a medical device's characteristics"""
    device_id: str
    supported_measurements: Set[VitalSign]
    measurement_frequencies: Dict[VitalSign, int]  # Frequency in seconds for each vital sign
    device_quality: float  # 0.0 to 1.0
    is_medical_grade: bool
    battery_drain_rate: float  # % per hour
    signal_variance: float
    delay_probability: float
    max_delay: int  # Maximum delay in seconds
    jitter: Dict[VitalSign, int]  # Jitter in seconds for each measurement type

    def generate_timestamps(self, vital_sign: VitalSign, 
                          start_time: pd.Timestamp,
                          end_time: pd.Timestamp) -> pd.DatetimeIndex:
        """Generate measurement timestamps with jitter for a specific vital sign"""
        if vital_sign not in self.supported_measurements:
            raise ValueError(f"Device {self.device_id} does not support {vital_sign}")
            
        frequency = self.measurement_frequencies[vital_sign]
        jitter = self.jitter[vital_sign]
        
        duration = (end_time - start_time).total_seconds()
        n_measurements = int(duration / frequency)
        
        base_timestamps = pd.date_range(
            start=start_time,
            periods=n_measurements,
            freq=f"{frequency}s"
        )
        
        jitter_values = np.random.uniform(-jitter, jitter, n_measurements)
        timestamps = base_timestamps + pd.to_timedelta(jitter_values, unit='s')
        
        return pd.DatetimeIndex(sorted(timestamps))
    

class DeviceType(str, Enum):
    MULTIMONITOR_PRO = "MULTIMONITOR_PRO"
    PORTABLE_MONITOR = "PORTABLE_MONITOR"
    SMARTWATCH_PREMIUM = "SMARTWATCH_PREMIUM"
    SMARTWATCH_BASIC = "SMARTWATCH_BASIC"
    TEMP_PATCH = "TEMP_PATCH"
    BP_CUFF_AUTO = "BP_CUFF_AUTO"
    CHEST_PATCH = "CHEST_PATCH"
    MANUAL_NURSE = "MANUAL_NURSE"

# Define available devices
AVAILABLE_DEVICES = {
    DeviceType.MULTIMONITOR_PRO: Device(
        device_id=DeviceType.MULTIMONITOR_PRO,
        supported_measurements={
            VitalSign.HEART_RATE,
            VitalSign.OXYGEN_SATURATION,
            VitalSign.BLOOD_PRESSURE_SYSTOLIC
        },
        measurement_frequencies={
            VitalSign.HEART_RATE: 5,            # Every 5 seconds
            VitalSign.OXYGEN_SATURATION: 5,     # Every 5 seconds
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: 900  # Every 15 minutes
        },
        device_quality=0.95,
        is_medical_grade=True,
        battery_drain_rate=0.5,
        signal_variance=0.1,
        delay_probability=0.01,
        max_delay=30,
        jitter={
            VitalSign.HEART_RATE: 2,
            VitalSign.OXYGEN_SATURATION: 2,
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: 60
        }
    ),
    DeviceType.PORTABLE_MONITOR: Device(
        device_id=DeviceType.PORTABLE_MONITOR,
        supported_measurements={
            VitalSign.HEART_RATE,
            VitalSign.OXYGEN_SATURATION,
            VitalSign.BLOOD_PRESSURE_SYSTOLIC
        },
        measurement_frequencies={
            VitalSign.HEART_RATE: 60,           # Every minute
            VitalSign.OXYGEN_SATURATION: 60,    # Every minute
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: 1800  # Every 30 minutes
        },
        device_quality=0.9,
        is_medical_grade=True,
        battery_drain_rate=1.0,
        signal_variance=0.15,
        delay_probability=0.02,
        max_delay=60,
        jitter={
            VitalSign.HEART_RATE: 5,
            VitalSign.OXYGEN_SATURATION: 5,
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: 120
        }
    ),
    DeviceType.SMARTWATCH_PREMIUM: Device(
        device_id=DeviceType.SMARTWATCH_PREMIUM,
        supported_measurements={
            VitalSign.HEART_RATE,
            VitalSign.OXYGEN_SATURATION,
            VitalSign.RESPIRATORY_RATE,
            VitalSign.BLOOD_PRESSURE_SYSTOLIC
        },
        measurement_frequencies={
            VitalSign.HEART_RATE: 60,           # Every minute
            VitalSign.OXYGEN_SATURATION: 300,   # Every 5 minutes
            VitalSign.RESPIRATORY_RATE: 300,     # Every 5 minutes
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: 300 # Every 5 minutes
        },
        device_quality=0.85,
        is_medical_grade=False,
        battery_drain_rate=1.5,
        signal_variance=0.2,
        delay_probability=0.03,
        max_delay=90,
        jitter={
            VitalSign.HEART_RATE: 10,
            VitalSign.OXYGEN_SATURATION: 20,
            VitalSign.RESPIRATORY_RATE: 20,
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: 20
        }
    ),
    DeviceType.SMARTWATCH_BASIC: Device(
        device_id=DeviceType.SMARTWATCH_BASIC,
        supported_measurements={
            VitalSign.HEART_RATE,
            VitalSign.OXYGEN_SATURATION
        },
        measurement_frequencies={
            VitalSign.HEART_RATE: 60,          # Every minute
            VitalSign.OXYGEN_SATURATION: 300   # Every 5 minutes
        },
        device_quality=0.75,
        is_medical_grade=False,
        battery_drain_rate=2.0,
        signal_variance=0.25,
        delay_probability=0.05,
        max_delay=120,
        jitter={
            VitalSign.HEART_RATE: 15,
            VitalSign.OXYGEN_SATURATION: 30
        }
    ),
    DeviceType.TEMP_PATCH: Device(
        device_id=DeviceType.TEMP_PATCH,
        supported_measurements={VitalSign.TEMPERATURE},
        measurement_frequencies={
            VitalSign.TEMPERATURE: 300  # Every 5 minutes
        },
        device_quality=0.9,
        is_medical_grade=True,
        battery_drain_rate=0.2,
        signal_variance=0.05,
        delay_probability=0.01,
        max_delay=30,
        jitter={
            VitalSign.TEMPERATURE: 10
        }
    ),
    DeviceType.BP_CUFF_AUTO: Device(
        device_id=DeviceType.BP_CUFF_AUTO,
        supported_measurements={
            VitalSign.BLOOD_PRESSURE_SYSTOLIC
        },
        measurement_frequencies={
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: 300  # Every 5 minutes
        },
        device_quality=0.85,
        is_medical_grade=True,
        battery_drain_rate=0.5,
        signal_variance=0.15,
        delay_probability=0.02,
        max_delay=60,
        jitter={
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: 180
        }
    ),
    DeviceType.CHEST_PATCH: Device(
        device_id=DeviceType.CHEST_PATCH,
        supported_measurements={
            VitalSign.HEART_RATE,
            VitalSign.RESPIRATORY_RATE
        },
        measurement_frequencies={
            VitalSign.HEART_RATE: 10,          # Every 10 seconds
            VitalSign.RESPIRATORY_RATE: 10     # Every 10 seconds
        },
        device_quality=0.92,
        is_medical_grade=True,
        battery_drain_rate=0.8,
        signal_variance=0.12,
        delay_probability=0.02,
        max_delay=45,
        jitter={
            VitalSign.HEART_RATE: 3,
            VitalSign.RESPIRATORY_RATE: 3
        }
    ),
    DeviceType.MANUAL_NURSE: Device(
        device_id=DeviceType.MANUAL_NURSE,
        supported_measurements={
            VitalSign.CONSCIOUSNESS,
            VitalSign.RESPIRATORY_RATE,
            VitalSign.TEMPERATURE
        },
        measurement_frequencies={
            VitalSign.CONSCIOUSNESS: 14400,    # Every 4 hours
            VitalSign.RESPIRATORY_RATE: 14400, # Every 4 hours
            VitalSign.TEMPERATURE: 14400      # Every 4 hours
        },
        device_quality=1.0,
        is_medical_grade=True,
        battery_drain_rate=0.0,
        signal_variance=0.0,
        delay_probability=0.2,
        max_delay=900,
        jitter={
            VitalSign.CONSCIOUSNESS: 1800,
            VitalSign.RESPIRATORY_RATE: 1800,
            VitalSign.TEMPERATURE: 1800
        }
    )
}