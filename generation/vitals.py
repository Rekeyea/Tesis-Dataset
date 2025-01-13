from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


class VitalSign(str, Enum):
    RESPIRATORY_RATE = "RESPIRATORY_RATE"
    OXYGEN_SATURATION = "OXYGEN_SATURATION"
    BLOOD_PRESSURE_SYSTOLIC = "BLOOD_PRESSURE_SYSTOLIC"
    HEART_RATE = "HEART_RATE"
    TEMPERATURE = "TEMPERATURE"
    CONSCIOUSNESS = "CONSCIOUSNESS"

@dataclass
class BaseVitalRange:
    """Base class for vital sign ranges"""
    base: float
    variance: float

    def _apply_random_walk(self, n: int, scale: float = 0.1) -> np.ndarray:
        """Generate random walk component"""
        if n > 1:
            random_walk = np.random.normal(0, self.variance, n)
            return np.cumsum(random_walk) * scale
        return np.zeros(n)

@dataclass
class RespiratoryRateRange(BaseVitalRange):
    """Respiratory rate specific range handling with NEWS2 alignment"""
    def generate_values(self, n: int, deterioration_probability: float = 0.0) -> np.ndarray:
        values = np.zeros(n)
        is_deteriorating = False
        current_base = self.base
        
        for i in range(n):
            # Check for new deterioration
            if not is_deteriorating and np.random.random() < deterioration_probability:
                is_deteriorating = True
                # In deterioration, RR typically increases by 4-8 breaths/min
                current_base = self.base + np.random.uniform(4, 8)
            
            # Generate the value for this timepoint
            random_walk = self._apply_random_walk(1, scale=0.05)[0]
            value = np.random.normal(current_base, self.variance) + random_walk
            
            # Apply NEWS2 aligned ranges
            if not is_deteriorating:
                value = np.clip(value, 12, 20)  # Normal range per NEWS2
            else:
                value = np.clip(value, 8, 35)  # Wider range for deterioration
                
            values[i] = value
            
        return np.round(values).astype(int)

@dataclass
class OxygenSaturationRange(BaseVitalRange):
    """Oxygen saturation specific range handling with NEWS2 alignment"""
    def generate_values(self, n: int, deterioration_probability: float = 0.0) -> np.ndarray:
        values = np.zeros(n)
        is_deteriorating = False
        current_base = self.base
        
        for i in range(n):
            # Check for new deterioration
            if not is_deteriorating and np.random.random() < deterioration_probability:
                is_deteriorating = True
                # SpO2 typically drops by 4-8% in deterioration
                current_base = self.base - np.random.uniform(4, 8)
            
            # Generate the value for this timepoint
            random_walk = self._apply_random_walk(1, scale=0.02)[0]
            value = np.random.normal(current_base, self.variance) + random_walk
            
            # Apply NEWS2 aligned ranges
            if not is_deteriorating:
                value = np.clip(value, 96, 100)  # Normal range
            else:
                value = np.clip(value, 83, 100)  # NEWS2 scale goes down to 83
                
            values[i] = value
            
        return np.round(values).astype(int)

@dataclass
class BloodPressureRange(BaseVitalRange):
    """Blood pressure specific range handling with NEWS2 alignment"""
    def generate_values(self, n: int, deterioration_probability: float = 0.0) -> np.ndarray:
        values = np.zeros(n)
        is_deteriorating = False
        current_base = self.base
        trend_direction = None
        
        for i in range(n):
            # Check for new deterioration
            if not is_deteriorating and np.random.random() < deterioration_probability:
                is_deteriorating = True
                # BP can go either way in deterioration
                trend_direction = np.random.choice([-1, 1])
                # Change by 15-25 mmHg in either direction
                change = np.random.uniform(15, 25)
                current_base = self.base + (change * trend_direction)
            
            # Generate the value for this timepoint
            random_walk = self._apply_random_walk(1, scale=0.05)[0]
            value = np.random.normal(current_base, self.variance) + random_walk
            
            # Apply NEWS2 aligned ranges
            if not is_deteriorating:
                value = np.clip(value, 110, 140)  # Normal range
            else:
                value = np.clip(value, 90, 220)  # NEWS2 considers <90 and >220 as extreme
                
            values[i] = value
            
        return np.round(values).astype(int)

@dataclass
class HeartRateRange(BaseVitalRange):
    """Heart rate specific range handling with NEWS2 alignment"""
    def generate_values(self, n: int, deterioration_probability: float = 0.0) -> np.ndarray:
        values = np.zeros(n)
        is_deteriorating = False
        current_base = self.base
        
        for i in range(n):
            # Check for new deterioration
            if not is_deteriorating and np.random.random() < deterioration_probability:
                is_deteriorating = True
                # When deteriorating, HR typically increases by 20-40 bpm
                current_base = self.base + np.random.uniform(20, 40)
            
            # Generate the value for this timepoint
            random_walk = self._apply_random_walk(1, scale=0.05)[0]
            value = np.random.normal(current_base, self.variance) + random_walk
            
            # Apply NEWS2 aligned ranges
            if not is_deteriorating:
                value = np.clip(value, 51, 90)  # Normal range per NEWS2
            else:
                value = np.clip(value, 40, 130)  # NEWS2 scale
                
            values[i] = value
            
        return np.round(values).astype(int)

@dataclass
class TemperatureRange(BaseVitalRange):
    """Temperature specific range handling with NEWS2 alignment"""
    def generate_values(self, n: int, deterioration_probability: float = 0.0) -> np.ndarray:
        values = np.zeros(n)
        is_deteriorating = False
        current_base = self.base
        trend_direction = None
        
        for i in range(n):
            # Check for new deterioration
            if not is_deteriorating and np.random.random() < deterioration_probability:
                is_deteriorating = True
                # Temperature can go either way in deterioration
                trend_direction = np.random.choice([-1, 1])
                # Change by 0.5-1.5°C in either direction
                change = np.random.uniform(0.5, 1.5)
                current_base = self.base + (change * trend_direction)
            
            # Generate the value for this timepoint
            random_walk = self._apply_random_walk(1, scale=0.05)[0]
            value = np.random.normal(current_base, self.variance) + random_walk
            
            # Apply NEWS2 aligned ranges
            if not is_deteriorating:
                value = np.clip(value, 36.1, 38.0)  # Normal range per NEWS2
            else:
                value = np.clip(value, 35.0, 39.5)  # NEWS2 scale
                
            values[i] = value
            
        return np.round(values * 10) / 10  # One decimal place

@dataclass
class ConsciousnessRange:
    """Consciousness level range handling"""
    base: str
    allowed_states: List[str]
    transition_probability: float = 0.0

    def generate_values(self, n: int, deterioration_probability: float = 1.0) -> np.ndarray:
        base_idx = self.allowed_states.index(self.base)
        values = np.full(n, self.base)
        
        if deterioration_probability > 1 and len(self.allowed_states) > 1:
            prob = self.transition_probability * (deterioration_probability - 1)
            transitions = np.random.random(n) < prob
            worse_state_idx = min(base_idx + 1, len(self.allowed_states) - 1)
            values[transitions] = self.allowed_states[worse_state_idx]
        
        return values