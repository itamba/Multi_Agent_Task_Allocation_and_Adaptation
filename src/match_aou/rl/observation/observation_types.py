"""
Data Structures for Observation Space

This module defines the core data structures used in observation extraction:
    - SelfState: 6 self features
    - TargetInfo: Information about a single target
    - ObservationOutput: Complete observation with metadata
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class SelfState:
    """
    Self features (6 total).
    
    All values are normalized to [0, 1] or binary {0, 1}.
    
    Attributes:
        fuel_norm: Current fuel / max fuel [0-1]
        has_weapon: Binary flag {0, 1} - has available weapons
        dist_to_next_step_norm: Distance to next waypoint / weapon_range [0-1]
        next_step_is_attack: Binary {0, 1} - is next action an attack
        rtb_possible: Binary {0, 1} - can safely return to base
        plan_progress: Steps completed / total steps [0-1]
    """
    
    fuel_norm: float
    has_weapon: float
    dist_to_next_step_norm: float
    next_step_is_attack: float
    rtb_possible: float
    plan_progress: float
    
    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array [6].
        
        Returns:
            np.array of shape (6,) with dtype float32
        """
        return np.array([
            self.fuel_norm,
            self.has_weapon,
            self.dist_to_next_step_norm,
            self.next_step_is_attack,
            self.rtb_possible,
            self.plan_progress
        ], dtype=np.float32)
    
    def validate(self) -> None:
        """
        Validate that all values are in valid range.
        
        Raises:
            ValueError: If any value is out of range
        """
        values = [
            ("fuel_norm", self.fuel_norm),
            ("has_weapon", self.has_weapon),
            ("dist_to_next_step_norm", self.dist_to_next_step_norm),
            ("next_step_is_attack", self.next_step_is_attack),
            ("rtb_possible", self.rtb_possible),
            ("plan_progress", self.plan_progress),
        ]
        
        for name, value in values:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass
class TargetInfo:
    """
    Information about a single target.
    
    Attributes:
        exists: Is this slot filled? (False for padding)
        id: Target unit ID
        latitude: Target latitude
        longitude: Target longitude
        distance_km: Distance from agent (km)
        is_threat: Can target attack agent?
        is_dynamic: Is target mobile (aircraft/ship vs facility)?
        is_in_plan: Is target already in agent's plan?
        side_id: Target's side ID (for filtering)
        already_engaged: Number of weapons currently targeting this unit
    """
    
    exists: bool
    id: str
    latitude: float
    longitude: float
    distance_km: float
    is_threat: bool
    is_dynamic: bool
    is_in_plan: bool
    side_id: str
    already_engaged: int = 0
    
    def to_array(self, weapon_range_km: float) -> np.ndarray:
        """
        Convert to normalized array [6].
        
        Args:
            weapon_range_km: Weapon engagement range for normalization
        
        Returns:
            np.array of shape (6,) with dtype float32
            Format: [exists, dist_norm, is_threat, is_dynamic, is_in_plan, already_engaged_norm]
        """
        if not self.exists:
            # Padding target - all zeros
            return np.zeros(6, dtype=np.float32)
        
        # Normalize distance by weapon range
        dist_norm = min(self.distance_km / max(weapon_range_km, 1.0), 1.0)
        
        # Normalize already_engaged by max expected value (3)
        # Typical scenarios: 0-3 weapons per target
        engaged_norm = min(self.already_engaged / 3.0, 1.0)
        
        return np.array([
            1.0,  # exists
            dist_norm,
            1.0 if self.is_threat else 0.0,
            1.0 if self.is_dynamic else 0.0,
            1.0 if self.is_in_plan else 0.0,
            engaged_norm  # NEW: engagement count
        ], dtype=np.float32)
    
    @staticmethod
    def create_empty() -> "TargetInfo":
        """
        Create an empty (padding) target.
        
        Returns:
            TargetInfo with exists=False and all other fields set to defaults
        """
        return TargetInfo(
            exists=False,
            id="",
            latitude=0.0,
            longitude=0.0,
            distance_km=0.0,
            is_threat=False,
            is_dynamic=False,
            is_in_plan=False,
            side_id="",
            already_engaged=0
        )


@dataclass
class ObservationOutput:
    """
    Complete observation with metadata.
    
    This is the output of build_observation_vector().
    
    Attributes:
        vector: Feature vector normalized to [0, 1]
            Size: 6 + (K × 6) + 6 where K = config.top_k
            - 6 self features
            - K × 6 target features (K targets × 6 features each)
            - 6 plan context features (optional, zeros if tasks/solution not provided)
            
            Example for top_k=3: [30] = 6 + 18 + 6
            Example for top_k=5: [42] = 6 + 30 + 6
            
        self_state: SelfState object (6 features)
        targets: List of TargetInfo objects (exactly K targets, padded if needed)
        agent_id: Agent's unique ID
        current_time: Simulation time (tick)
    """
    
    vector: np.ndarray
    self_state: SelfState
    targets: List[TargetInfo]
    agent_id: str
    current_time: int
    
    def validate(self, max_targets: int = None) -> None:
        """
        Validate observation integrity.
        
        Args:
            max_targets: Maximum number of targets allowed (optional, defaults to len(targets))
        
        Raises:
            AssertionError: If validation fails
        """
        # Check vector shape: 6 self + K×6 target features + 6 plan context
        # Can be either [24] (legacy, no plan context) or [30] (with plan context)
        expected_size_legacy = 6 + len(self.targets) * 6  # [24]
        expected_size_new = 6 + len(self.targets) * 6 + 6  # [30]
        
        assert self.vector.shape[0] in [expected_size_legacy, expected_size_new], \
            f"Expected shape ({expected_size_legacy},) or ({expected_size_new},), got {self.vector.shape}"
        
        # Check vector values in [0, 1]
        assert np.all((self.vector >= 0) & (self.vector <= 1)), \
            f"Vector values must be in [0, 1], got min={self.vector.min()}, max={self.vector.max()}"
        
        # Check number of targets (if max_targets specified)
        if max_targets is not None:
            assert len(self.targets) <= max_targets, \
                f"Too many targets: {len(self.targets)} > {max_targets}"
        
        # Check self state
        self.self_state.validate()
