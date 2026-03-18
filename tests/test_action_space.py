"""
Test Suite for Action Space Module

Tests all components of the action space:
- Action configuration
- Action validation
- Action utilities
- API functions

Run: python test_action_space.py
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


# ============================================================================
# Mock Classes (since we don't have full project in test environment)
# ============================================================================

@dataclass
class MockSelfState:
    """Mock self state for testing."""
    fuel_norm: float = 0.75
    has_weapon: float = 1.0
    dist_to_next_step_norm: float = 0.5
    next_step_is_attack: float = 0.0
    rtb_possible: float = 1.0
    plan_progress: float = 0.3


@dataclass
class MockTargetInfo:
    """Mock target info for testing."""
    exists: bool = True
    id: str = "target_01"
    latitude: float = 35.5
    longitude: float = 40.5
    distance_km: float = 50.0
    is_threat: bool = False
    is_dynamic: bool = True
    is_in_plan: bool = False
    side_id: str = "red"
    already_engaged: int = 0


@dataclass
class MockObservationOutput:
    """Mock observation output for testing."""
    vector: np.ndarray = None
    self_state: MockSelfState = None
    targets: List[MockTargetInfo] = None
    agent_id: str = "f16_01"
    current_time: int = 100
    
    def __post_init__(self):
        if self.self_state is None:
            self.self_state = MockSelfState()
        if self.targets is None:
            self.targets = [MockTargetInfo(), MockTargetInfo(), MockTargetInfo()]


class MockAircraft:
    """Mock aircraft for testing."""
    def __init__(self):
        self.id = "f16_01"
        self.latitude = 35.0
        self.longitude = 40.0
        self.rtb = False
        self.home_base_id = "base_01"
        self.current_fuel = 7500
        self.max_fuel = 10000


class MockAirbase:
    """Mock airbase for testing."""
    def __init__(self):
        self.id = "base_01"
        self.latitude = 35.1
        self.longitude = 40.1


class MockScenario:
    """Mock BLADE scenario for testing."""
    def __init__(self):
        self.aircraft_dict = {"f16_01": MockAircraft()}
        self.airbase_dict = {"base_01": MockAirbase()}
    
    def get_aircraft(self, aircraft_id):
        return self.aircraft_dict.get(aircraft_id)
    
    def get_airbase(self, airbase_id):
        return self.airbase_dict.get(airbase_id)
    
    def get_ship(self, ship_id):
        return None


# ============================================================================
# Import module to test
# ============================================================================

# Add parent directory to path
sys.path.insert(0, '/home/claude')

from rl_action_space import (
    ActionSpaceConfig,
    ActionType,
    compute_action_mask,
    validate_action,
    action_to_string,
    string_to_action,
    get_action_space_size,
    action_index_to_target_slot,
    is_attack_action,
    is_noop_action,
    is_rtb_action
)


# ============================================================================
# Tests
# ============================================================================

def test_action_config():
    """Test action space configuration."""
    print("\nTest: Action Configuration")
    
    # Default config
    config = ActionSpaceConfig()
    assert config.top_k == 3
    assert config.enable_rtb == True
    assert config.enable_noop == True
    
    # Action space size
    assert config.get_action_space_size() == 5  # NOOP + 3 attacks + RTB
    
    # Custom config
    config2 = ActionSpaceConfig(top_k=2, enable_rtb=False)
    assert config2.get_action_space_size() == 3  # NOOP + 2 attacks
    
    # Action to string
    assert config.action_to_string(ActionType.NOOP) == "NOOP"
    assert config.action_to_string(ActionType.INSERT_ATTACK_0) == "INSERT_ATTACK(0)"
    assert config.action_to_string(ActionType.INSERT_ATTACK_1) == "INSERT_ATTACK(1)"
    assert config.action_to_string(ActionType.FORCE_RTB) == "FORCE_RTB"
    
    # String to action
    assert config.string_to_action("NOOP") == ActionType.NOOP
    assert config.string_to_action("INSERT_ATTACK(0)") == ActionType.INSERT_ATTACK_0
    assert config.string_to_action("INSERT_ATTACK(1)") == ActionType.INSERT_ATTACK_1
    assert config.string_to_action("FORCE_RTB") == ActionType.FORCE_RTB
    assert config.string_to_action("INVALID") is None
    
    print("✓ Action configuration tests passed")


def test_action_validation_basic():
    """Test basic action validation."""
    print("\nTest: Basic Action Validation")
    
    config = ActionSpaceConfig(top_k=3)
    obs = MockObservationOutput()
    scenario = MockScenario()
    
    # Compute action mask
    mask = compute_action_mask(obs, scenario, "f16_01", config)
    
    # Check mask structure
    assert len(mask.mask) == 5  # NOOP + 3 attacks + RTB
    assert mask.mask[ActionType.NOOP] == 1  # NOOP always valid
    
    # Check valid actions can be extracted
    valid_actions = mask.get_valid_actions()
    assert ActionType.NOOP in valid_actions
    
    print(f"  Valid actions: {[action_to_string(a, config) for a in valid_actions]}")
    print("✓ Basic validation tests passed")


def test_action_validation_no_weapons():
    """Test validation when agent has no weapons."""
    print("\nTest: No Weapons Validation")
    
    config = ActionSpaceConfig(top_k=3)
    obs = MockObservationOutput()
    obs.self_state.has_weapon = 0.0  # No weapons
    scenario = MockScenario()
    
    mask = compute_action_mask(obs, scenario, "f16_01", config)
    
    # NOOP should be valid
    assert mask.is_valid(ActionType.NOOP)
    
    # All INSERT_ATTACK should be invalid
    assert not mask.is_valid(ActionType.INSERT_ATTACK_0)
    assert not mask.is_valid(ActionType.INSERT_ATTACK_1)
    assert not mask.is_valid(ActionType.INSERT_ATTACK_2)
    
    # Check reasons
    reason = mask.get_invalid_reason(ActionType.INSERT_ATTACK_0)
    assert "weapon" in reason.lower()
    
    print(f"  Reason for invalid attack: {reason}")
    print("✓ No weapons validation passed")


def test_action_validation_low_fuel():
    """Test validation when fuel is too low."""
    print("\nTest: Low Fuel Validation")
    
    config = ActionSpaceConfig(top_k=3, min_attack_fuel_margin=0.3)
    obs = MockObservationOutput()
    obs.self_state.fuel_norm = 0.2  # Below 30% threshold
    scenario = MockScenario()
    
    mask = compute_action_mask(obs, scenario, "f16_01", config)
    
    # INSERT_ATTACK should be invalid
    assert not mask.is_valid(ActionType.INSERT_ATTACK_0)
    
    reason = mask.get_invalid_reason(ActionType.INSERT_ATTACK_0)
    assert "fuel" in reason.lower()
    
    print(f"  Reason: {reason}")
    print("✓ Low fuel validation passed")


def test_action_validation_target_in_plan():
    """Test validation when target already in plan."""
    print("\nTest: Target In Plan Validation")
    
    config = ActionSpaceConfig(top_k=3)
    obs = MockObservationOutput()
    obs.targets[0].is_in_plan = True  # Target 0 already planned
    scenario = MockScenario()
    
    mask = compute_action_mask(obs, scenario, "f16_01", config)
    
    # INSERT_ATTACK(0) should be invalid
    assert not mask.is_valid(ActionType.INSERT_ATTACK_0)
    
    # INSERT_ATTACK(1) should be valid (target 1 not in plan)
    assert mask.is_valid(ActionType.INSERT_ATTACK_1)
    
    reason = mask.get_invalid_reason(ActionType.INSERT_ATTACK_0)
    assert "already in plan" in reason.lower()
    
    print(f"  Reason: {reason}")
    print("✓ Target in plan validation passed")


def test_action_validation_cooldown():
    """Test attack cooldown validation."""
    print("\nTest: Attack Cooldown Validation")
    
    config = ActionSpaceConfig(top_k=3, attack_cooldown_ticks=50)
    obs = MockObservationOutput()
    obs.current_time = 100
    scenario = MockScenario()
    
    # Last attack was 30 ticks ago (still in cooldown)
    last_attack_tick = 70
    mask = compute_action_mask(obs, scenario, "f16_01", config, last_attack_tick)
    
    # All attacks should be invalid
    assert not mask.is_valid(ActionType.INSERT_ATTACK_0)
    
    reason = mask.get_invalid_reason(ActionType.INSERT_ATTACK_0)
    assert "cooldown" in reason.lower()
    
    print(f"  Reason: {reason}")
    
    # No cooldown if last_attack_tick is None
    mask2 = compute_action_mask(obs, scenario, "f16_01", config, None)
    assert mask2.is_valid(ActionType.INSERT_ATTACK_0)
    
    print("✓ Cooldown validation passed")


def test_action_validation_rtb():
    """Test RTB validation."""
    print("\nTest: RTB Validation")
    
    config = ActionSpaceConfig(top_k=3, enable_rtb=True)
    obs = MockObservationOutput()
    scenario = MockScenario()
    
    # Normal case - RTB should be valid
    mask = compute_action_mask(obs, scenario, "f16_01", config)
    assert mask.is_valid(ActionType.FORCE_RTB)
    
    # Already RTB - should be invalid
    scenario.aircraft_dict["f16_01"].rtb = True
    mask2 = compute_action_mask(obs, scenario, "f16_01", config)
    assert not mask2.is_valid(ActionType.FORCE_RTB)
    
    reason = mask2.get_invalid_reason(ActionType.FORCE_RTB)
    assert "already" in reason.lower()
    
    print(f"  Reason when already RTB: {reason}")
    print("✓ RTB validation passed")


def test_action_utilities():
    """Test utility functions."""
    print("\nTest: Action Utilities")
    
    # Target slot extraction
    assert action_index_to_target_slot(ActionType.INSERT_ATTACK_0) == 0
    assert action_index_to_target_slot(ActionType.INSERT_ATTACK_1) == 1
    assert action_index_to_target_slot(ActionType.INSERT_ATTACK_2) == 2
    assert action_index_to_target_slot(ActionType.NOOP) is None
    assert action_index_to_target_slot(ActionType.FORCE_RTB) is None
    
    # Action type checks
    assert is_attack_action(ActionType.INSERT_ATTACK_0)
    assert is_attack_action(ActionType.INSERT_ATTACK_1)
    assert not is_attack_action(ActionType.NOOP)
    assert not is_attack_action(ActionType.FORCE_RTB)
    
    assert is_noop_action(ActionType.NOOP)
    assert not is_noop_action(ActionType.INSERT_ATTACK_0)
    
    assert is_rtb_action(ActionType.FORCE_RTB)
    assert not is_rtb_action(ActionType.NOOP)
    
    print("✓ Utility function tests passed")


def test_api_functions():
    """Test public API functions."""
    print("\nTest: API Functions")
    
    config = ActionSpaceConfig(top_k=3)
    
    # Action space size
    assert get_action_space_size(config) == 5
    
    # Action conversion
    assert action_to_string(0, config) == "NOOP"
    assert action_to_string(1, config) == "INSERT_ATTACK(0)"
    assert action_to_string(4, config) == "FORCE_RTB"
    
    assert string_to_action("NOOP", config) == 0
    assert string_to_action("INSERT_ATTACK(1)", config) == 2
    assert string_to_action("FORCE_RTB", config) == 4
    
    # Single action validation
    obs = MockObservationOutput()
    scenario = MockScenario()
    
    is_valid, reason = validate_action(ActionType.NOOP, obs, scenario, "f16_01", config)
    assert is_valid
    assert reason is None
    
    # Invalid action
    obs.self_state.has_weapon = 0.0
    is_valid, reason = validate_action(ActionType.INSERT_ATTACK_0, obs, scenario, "f16_01", config)
    assert not is_valid
    assert reason is not None
    
    print("✓ API function tests passed")


def test_edge_cases():
    """Test edge cases."""
    print("\nTest: Edge Cases")
    
    config = ActionSpaceConfig(top_k=3)
    
    # Agent not found
    obs = MockObservationOutput()
    scenario = MockScenario()
    scenario.aircraft_dict = {}  # Empty
    
    mask = compute_action_mask(obs, scenario, "f16_01", config)
    
    # All actions should be invalid
    assert all(m == 0 for m in mask.mask)
    
    # No targets exist
    obs2 = MockObservationOutput()
    obs2.targets = [
        MockTargetInfo(exists=False),
        MockTargetInfo(exists=False),
        MockTargetInfo(exists=False)
    ]
    scenario2 = MockScenario()
    
    mask2 = compute_action_mask(obs2, scenario2, "f16_01", config)
    
    # NOOP and RTB valid, attacks invalid
    assert mask2.is_valid(ActionType.NOOP)
    assert not mask2.is_valid(ActionType.INSERT_ATTACK_0)
    
    print("✓ Edge case tests passed")


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all test functions."""
    print("=" * 70)
    print("ACTION SPACE MODULE - TEST SUITE")
    print("=" * 70)
    
    try:
        test_action_config()
        test_action_validation_basic()
        test_action_validation_no_weapons()
        test_action_validation_low_fuel()
        test_action_validation_target_in_plan()
        test_action_validation_cooldown()
        test_action_validation_rtb()
        test_action_utilities()
        test_api_functions()
        test_edge_cases()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
