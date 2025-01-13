from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from generation.devices import Device
from generation.vitals import BaseVitalRange, BloodPressureRange, ConsciousnessRange, HeartRateRange, OxygenSaturationRange, RespiratoryRateRange, TemperatureRange, VitalSign

@dataclass
class PatientTemplate:
    """Defines expected vital ranges for a patient condition"""
    template_id: str
    vital_ranges: Dict[VitalSign, BaseVitalRange]
    deterioration_probability: float

@dataclass
class Patient:
    """Represents a patient with assigned devices and a template"""
    patient_id: str
    template: PatientTemplate
    assigned_devices: List[Device]

class PatientTemplateType(str, Enum):
    HEALTHY = "HEALTHY"
    STABLE_CHRONIC = "STABLE_CHRONIC"
    DETERIORATING = "DETERIORATING"

# Define patient templates
PATIENT_TEMPLATES = {
    PatientTemplateType.HEALTHY: PatientTemplate(
        template_id=PatientTemplateType.HEALTHY,
        vital_ranges={
            VitalSign.RESPIRATORY_RATE: RespiratoryRateRange(16, 4),      # 12-20 normal
            VitalSign.OXYGEN_SATURATION: OxygenSaturationRange(99, 1),  # 96-100 normal
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: BloodPressureRange(105, 10),
            VitalSign.HEART_RATE: HeartRateRange(75, 10),                  # 51-90 normal
            VitalSign.TEMPERATURE: TemperatureRange(36.5, 0.5),           # 36.1-38.0 normal
            VitalSign.CONSCIOUSNESS: ConsciousnessRange("A", ["A"])       # Alert only
        },
        deterioration_probability=0
    ),   
    PatientTemplateType.STABLE_CHRONIC: PatientTemplate(
        template_id=PatientTemplateType.STABLE_CHRONIC,
        vital_ranges={
            VitalSign.RESPIRATORY_RATE: RespiratoryRateRange(20, 3),    # Slightly elevated
            VitalSign.OXYGEN_SATURATION: OxygenSaturationRange(95, 1),    # Lower normal
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: BloodPressureRange(150, 20),
            VitalSign.HEART_RATE: HeartRateRange(85, 10),                  # Upper normal
            VitalSign.TEMPERATURE: TemperatureRange(36.5, 0.5),           # Normal
            VitalSign.CONSCIOUSNESS: ConsciousnessRange("A", ["A", "V"], 0.01)
        },
        deterioration_probability=0.01
    ),   
    PatientTemplateType.DETERIORATING: PatientTemplate(
        template_id=PatientTemplateType.DETERIORATING,
        vital_ranges={
            VitalSign.RESPIRATORY_RATE: RespiratoryRateRange(16, 4),      # 12-20 normal
            VitalSign.OXYGEN_SATURATION: OxygenSaturationRange(99, 1),  # 96-100 normal
            VitalSign.BLOOD_PRESSURE_SYSTOLIC: BloodPressureRange(105, 10),
            VitalSign.HEART_RATE: HeartRateRange(75, 10),                  # 51-90 normal
            VitalSign.TEMPERATURE: TemperatureRange(36.5, 0.5),           # 36.1-38.0 normal
            VitalSign.CONSCIOUSNESS: ConsciousnessRange("A", ["A", "V", "P"])
        },
        deterioration_probability=0.1
    )
}