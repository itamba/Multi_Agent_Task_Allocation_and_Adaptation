"""
Action Validation

Checks whether actions are legal given current state.

Validation Rules:
- INSERT_ATTACK: Target exists, has weapon, enough fuel, not in cooldown
- FORCE_RTB: Not already RTB, far enough from base, has base
- NOOP: Always valid

Integration:
- Uses observation data to check legality
- Respects BLADE constraints (FIFO, fuel, etc.)
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from .action_config import ActionType, ActionSpaceConfig


@dataclass(frozen=True)
class ActionMask:
    """Binary mask of valid actions.
    
    Attributes:
        mask: List of 0/1 indicating which actions are valid
        reasons: Dict mapping action index to reason why invalid
    """
    mask: List[int]  # 0 = invalid, 1 = valid
    reasons: Dict[int, str]  # action_idx -> reason for invalidity
    
    def is_valid(self, action: int) -> bool:
        """Check if action is valid."""
        if 0 <= action < len(self.mask):
            return self.mask[action] == 1
        return False
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices."""
        return [i for i, valid in enumerate(self.mask) if valid == 1]
    
    def get_invalid_reason(self, action: int) -> Optional[str]:
        """Get reason why action is invalid."""
        return self.reasons.get(action)


class ActionValidator:
    """Validates actions against current state.
    
    Uses observation data to determine which actions are legal.
    """
    
    def __init__(self, config: ActionSpaceConfig):
        """Initialize validator.
        
        Args:
            config: Action space configuration
        """
        self.config = config
    
    def validate_action_mask(
        self,
        observation_output,  # ObservationOutput from Phase 1
        scenario,            # BLADE Scenario
        agent_id: str,
        last_attack_tick: Optional[int] = None
    ) -> ActionMask:
        """Compute action mask for current state.
        
        Args:
            observation_output: ObservationOutput from build_observation_vector
            scenario: BLADE Scenario object
            agent_id: Agent ID
            last_attack_tick: Tick of last attack (for cooldown)
            
        Returns:
            ActionMask with valid/invalid flags and reasons
        """
        mask = []
        reasons = {}
        
        # Get state from observation
        self_state = observation_output.self_state
        targets = observation_output.targets
        current_time = observation_output.current_time
        
        # Get agent from scenario
        aircraft = scenario.get_aircraft(agent_id)
        if aircraft is None:
            # If agent not found, all actions invalid
            return ActionMask(
                mask=[0] * self.config.get_action_space_size(),
                reasons={i: "Agent not found" for i in range(self.config.get_action_space_size())}
            )
        
        # === Validate NOOP ===
        if self.config.enable_noop:
            # NOOP always valid
            mask.append(1)
        
        # === Validate INSERT_ATTACK(k) for each target ===
        for k in range(self.config.top_k):
            action_idx = ActionType.INSERT_ATTACK_0 + k
            
            # Check if target exists
            if k >= len(targets) or not targets[k].exists:
                mask.append(0)
                reasons[action_idx] = f"Target {k} does not exist"
                continue
            
            target = targets[k]
            
            # Check if already in plan
            if target.is_in_plan:
                mask.append(0)
                reasons[action_idx] = f"Target {k} already in plan"
                continue
            
            # Check if agent has weapons
            if self_state.has_weapon < 0.5:  # Binary threshold
                mask.append(0)
                reasons[action_idx] = "No weapons available"
                continue
            
            # Check fuel margin
            if self_state.fuel_norm < self.config.min_attack_fuel_margin:
                mask.append(0)
                reasons[action_idx] = (
                    f"Insufficient fuel (need {self.config.min_attack_fuel_margin:.1%}, "
                    f"have {self_state.fuel_norm:.1%})"
                )
                continue
            
            # Check attack cooldown
            if last_attack_tick is not None:
                ticks_since_attack = current_time - last_attack_tick
                if ticks_since_attack < self.config.attack_cooldown_ticks:
                    mask.append(0)
                    reasons[action_idx] = (
                        f"Attack cooldown (wait {self.config.attack_cooldown_ticks - ticks_since_attack} ticks)"
                    )
                    continue
            
            # All checks passed
            mask.append(1)
        
        # === Validate FORCE_RTB ===
        if self.config.enable_rtb:
            action_idx = ActionType.FORCE_RTB
            
            # Check if already RTB
            if getattr(aircraft, 'rtb', False):
                mask.append(0)
                reasons[action_idx] = "Already returning to base"
            
            # Check if RTB possible
            elif self_state.rtb_possible < 0.5:  # Binary threshold
                mask.append(0)
                reasons[action_idx] = "RTB not possible (insufficient fuel)"
            
            # Check distance from base (avoid spam)
            elif not self._is_far_from_base(aircraft, scenario):
                mask.append(0)
                reasons[action_idx] = (
                    f"Too close to base (need >{self.config.min_rtb_distance_km:.1f}km)"
                )
            
            else:
                # RTB valid
                mask.append(1)
        
        return ActionMask(mask=mask, reasons=reasons)
    
    def _is_far_from_base(self, aircraft, scenario) -> bool:
        """Check if aircraft is far enough from base to issue RTB.
        
        Args:
            aircraft: Aircraft object
            scenario: BLADE Scenario
            
        Returns:
            True if distance > min_rtb_distance_km
        """
        try:
            # Get home base
            home_base_id = getattr(aircraft, 'home_base_id', None)
            if not home_base_id:
                return True  # No home base, allow RTB
            
            # Get base location
            home_base = None
            if hasattr(scenario, 'get_airbase'):
                home_base = scenario.get_airbase(home_base_id)
            
            if not home_base and hasattr(scenario, 'get_ship'):
                home_base = scenario.get_ship(home_base_id)
            
            if not home_base:
                return True  # Base not found, allow RTB
            
            # Compute distance using shared utility
            from ..shared_utils import haversine_distance
            
            distance_km = haversine_distance(
                (aircraft.latitude, aircraft.longitude),
                (home_base.latitude, home_base.longitude)
            )
            
            return distance_km > self.config.min_rtb_distance_km
            
        except Exception:
            # On error, be conservative and allow RTB
            return True
    
    def validate_single_action(
        self,
        action: int,
        observation_output,
        scenario,
        agent_id: str,
        last_attack_tick: Optional[int] = None
    ) -> tuple[bool, Optional[str]]:
        """Validate a single action.
        
        Args:
            action: Action index to validate
            observation_output: ObservationOutput
            scenario: BLADE Scenario
            agent_id: Agent ID
            last_attack_tick: Last attack tick
            
        Returns:
            (is_valid, reason) tuple
        """
        mask = self.validate_action_mask(
            observation_output,
            scenario,
            agent_id,
            last_attack_tick
        )
        
        is_valid = mask.is_valid(action)
        reason = None if is_valid else mask.get_invalid_reason(action)
        
        return is_valid, reason
