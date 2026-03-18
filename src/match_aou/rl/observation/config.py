"""
Configuration for Observation Extraction

This module defines the configuration parameters for extracting
observations from BLADE scenarios.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ObservationConfig:
    """
    Configuration for observation extraction.
    
    Attributes:
        top_k: Number of target slots (default: 3)
        min_weapon_range_km: Fallback range if no weapons (default: 1.0 km)
        rtb_fuel_margin: Safety margin for RTB calculation (default: 1.2 = 20%)
        attack_action_keywords: Keywords to identify attack actions in plan
    """
    
    # Target filtering
    top_k: int = 3
    
    # Range limits
    min_weapon_range_km: float = 1.0
    
    # RTB safety margin (20% extra fuel)
    rtb_fuel_margin: float = 1.2
    
    # Plan parsing keywords
    attack_action_keywords: List[str] = field(default_factory=lambda: [
        "handle_aircraft_attack",
        "handle_ship_attack",
        "launch_weapon"
    ])
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        
        if self.min_weapon_range_km <= 0:
            raise ValueError(
                f"min_weapon_range_km must be positive, got {self.min_weapon_range_km}"
            )
        
        if self.rtb_fuel_margin < 1.0:
            raise ValueError(
                f"rtb_fuel_margin must be >= 1.0, got {self.rtb_fuel_margin}"
            )
        
        if not self.attack_action_keywords:
            raise ValueError("attack_action_keywords cannot be empty")
