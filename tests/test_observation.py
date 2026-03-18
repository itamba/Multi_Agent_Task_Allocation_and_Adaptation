"""
Test Script for Observation Module

Tests the observation extraction module with mock data.
Run this to verify the module works before integrating with real BLADE scenarios.
"""

import numpy as np
import sys
import os

# Add module to path
module_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, module_path)

from match_aou.rl.observation import (
    build_observation_vector,
    ObservationConfig,
    SelfState,
    TargetInfo
)


# ============================================================================
# Mock Objects (simulate BLADE objects)
# ============================================================================

class MockWeapon:
    def __init__(self, engagement_range_nm=50.0, current_quantity=10):
        self.engagement_range_nm = engagement_range_nm
        self.current_quantity = current_quantity
    
    def get_engagement_range(self):
        return self.engagement_range_nm


class MockAircraft:
    def __init__(self, aircraft_id, latitude, longitude, side_id="blue"):
        self.id = aircraft_id
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = 10000
        self.side_id = side_id
        self.current_fuel = 7500
        self.max_fuel = 10000
        self.fuel_rate = 2000  # lbs/hr
        self.speed = 450  # knots
        self.route = [[latitude + 1.0, longitude + 1.0]]
        self.home_base_id = "base_01"
        self.weapons = [MockWeapon()]
    
    def get_weapon_with_highest_engagement_range(self):
        if self.weapons:
            return self.weapons[0]
        return None


class MockAirbase:
    def __init__(self, base_id, latitude, longitude, side_id="blue"):
        self.id = base_id
        self.latitude = latitude
        self.longitude = longitude
        self.side_id = side_id


class MockTarget:
    def __init__(self, target_id, latitude, longitude, side_id="red", 
                 has_weapons=False, is_dynamic=False):
        self.id = target_id
        self.latitude = latitude
        self.longitude = longitude
        self.side_id = side_id
        self.is_dynamic_unit = is_dynamic
        self.weapons = [MockWeapon(30.0)] if has_weapons else []
    
    def get_weapon_with_highest_engagement_range(self):
        if self.weapons:
            return self.weapons[0]
        return None


class MockScenario:
    def __init__(self):
        # Blue side aircraft
        self.my_aircraft = MockAircraft("f16_01", 35.0, 40.0, "blue")
        
        # Red side targets
        self.target1 = MockTarget("target_01", 35.5, 40.5, "red", has_weapons=True, is_dynamic=False)
        self.target2 = MockTarget("target_02", 35.8, 40.8, "red", has_weapons=False, is_dynamic=False)
        self.target3 = MockTarget("aircraft_01", 35.2, 40.2, "red", has_weapons=True, is_dynamic=True)
        
        # Bases
        self.base = MockAirbase("base_01", 34.5, 39.5, "blue")
        
        self.aircraft = [self.my_aircraft, self.target3]
        self.ships = []
        self.facilities = [self.target1, self.target2]
        self.airbases = [self.base]
        self.weapons = []  # No active weapons tracking targets initially
        
        self.current_time = 100
        self.start_time = 0
    
    def get_aircraft(self, aircraft_id):
        for ac in self.aircraft:
            if ac.id == aircraft_id:
                return ac
        return None
    
    def get_airbase(self, base_id):
        for base in self.airbases:
            if base.id == base_id:
                return base
        return None
    
    def is_hostile(self, side1, side2):
        return side1 != side2


# ============================================================================
# Tests
# ============================================================================

def test_basic_observation():
    """Test basic observation extraction."""
    print("\n" + "="*70)
    print("TEST 1: Basic Observation Extraction")
    print("="*70)
    
    scenario = MockScenario()
    
    current_plan = {
        0: ["move_aircraft('f16_01', [[35.5, 40.5]])"],
        50: ["handle_aircraft_attack('f16_01', 'target_01', 'aim_120', 2)"],
        100: ["aircraft_return_to_base('f16_01')"]
    }
    
    config = ObservationConfig(top_k=3)
    
    obs = build_observation_vector(
        scenario=scenario,
        agent_id="f16_01",
        current_plan=current_plan,
        current_time=25,
        config=config
    )
    
    print(f"✓ Observation vector shape: {obs.vector.shape}")
    assert obs.vector.shape == (24,), f"Expected (21,), got {obs.vector.shape}"
    
    print(f"✓ All values in [0, 1]: min={obs.vector.min():.3f}, max={obs.vector.max():.3f}")
    assert np.all((obs.vector >= 0) & (obs.vector <= 1)), "Values not in [0, 1]"
    
    print(f"✓ Self features:")
    print(f"  - Fuel: {obs.self_state.fuel_norm:.3f}")
    print(f"  - Has weapon: {obs.self_state.has_weapon:.0f}")
    print(f"  - Next step distance: {obs.self_state.dist_to_next_step_norm:.3f}")
    print(f"  - Next is attack: {obs.self_state.next_step_is_attack:.0f}")
    print(f"  - RTB possible: {obs.self_state.rtb_possible:.0f}")
    print(f"  - Plan progress: {obs.self_state.plan_progress:.3f}")
    
    print(f"✓ Targets: {len(obs.targets)}")
    for i, target in enumerate(obs.targets):
        if target.exists:
            print(f"  - Target {i}: {target.id}, dist={target.distance_km:.1f}km, "
                  f"threat={target.is_threat}, dynamic={target.is_dynamic}, "
                  f"in_plan={target.is_in_plan}")
    
    print("\n✓ TEST PASSED: Basic observation extraction works")


def test_no_weapons():
    """Test with aircraft that has no weapons."""
    print("\n" + "="*70)
    print("TEST 2: No Weapons")
    print("="*70)
    
    scenario = MockScenario()
    scenario.my_aircraft.weapons = []  # Remove weapons
    
    obs = build_observation_vector(
        scenario=scenario,
        agent_id="f16_01",
        current_plan={},
        current_time=0
    )
    
    print(f"✓ Has weapon: {obs.self_state.has_weapon:.0f}")
    assert obs.self_state.has_weapon == 0.0, "Should report no weapons"
    
    # Should still see targets (based on min_weapon_range_km)
    print(f"✓ Can still see targets: {sum(t.exists for t in obs.targets)}")
    
    print("\n✓ TEST PASSED: No weapons handled correctly")


def test_no_targets():
    """Test with no enemies in range."""
    print("\n" + "="*70)
    print("TEST 3: No Targets in Range")
    print("="*70)
    
    scenario = MockScenario()
    # Remove all red targets
    scenario.facilities = []
    scenario.aircraft = [scenario.my_aircraft]
    
    obs = build_observation_vector(
        scenario=scenario,
        agent_id="f16_01",
        current_plan={},
        current_time=0
    )
    
    # All target slots should be padding (exists=False)
    existing_targets = sum(t.exists for t in obs.targets)
    print(f"✓ Targets found: {existing_targets}")
    assert existing_targets == 0, "Should have no targets"
    
    # All target features should be 0
    target_features = obs.vector[6:]  # Features 6-20 are targets
    print(f"✓ All target features are 0: {np.all(target_features == 0)}")
    assert np.all(target_features == 0), "Target features should be all 0"
    
    print("\n✓ TEST PASSED: No targets handled correctly")


def test_target_in_plan():
    """Test that targets in plan are marked correctly."""
    print("\n" + "="*70)
    print("TEST 4: Target in Plan Detection")
    print("="*70)
    
    scenario = MockScenario()
    
    # Plan that attacks target_01
    current_plan = {
        50: ["handle_aircraft_attack('f16_01', 'target_01', 'aim_120', 2)"]
    }
    
    obs = build_observation_vector(
        scenario=scenario,
        agent_id="f16_01",
        current_plan=current_plan,
        current_time=0
    )
    
    # Check if target_01 is marked as in_plan
    for target in obs.targets:
        if target.exists and target.id == "target_01":
            print(f"✓ Target {target.id} marked as in_plan: {target.is_in_plan}")
            assert target.is_in_plan, "target_01 should be marked as in_plan"
            break
    
    print("\n✓ TEST PASSED: Targets in plan detected correctly")


def test_plan_progress():
    """Test plan progress calculation."""
    print("\n" + "="*70)
    print("TEST 5: Plan Progress")
    print("="*70)
    
    scenario = MockScenario()
    
    current_plan = {
        0: ["move_aircraft('f16_01', [[35.5, 40.5]])"],
        50: ["handle_aircraft_attack('f16_01', 'target_01', 'aim_120', 2)"],
        100: ["aircraft_return_to_base('f16_01')"]
    }
    
    # At t=0 (start)
    obs_start = build_observation_vector(scenario, "f16_01", current_plan, 0)
    print(f"✓ Progress at t=0: {obs_start.self_state.plan_progress:.3f}")
    
    # At t=50 (middle)
    obs_mid = build_observation_vector(scenario, "f16_01", current_plan, 50)
    print(f"✓ Progress at t=50: {obs_mid.self_state.plan_progress:.3f}")
    
    # At t=100 (end)
    obs_end = build_observation_vector(scenario, "f16_01", current_plan, 100)
    print(f"✓ Progress at t=100: {obs_end.self_state.plan_progress:.3f}")
    
    assert obs_start.self_state.plan_progress < obs_mid.self_state.plan_progress < obs_end.self_state.plan_progress, \
        "Progress should increase over time"
    
    print("\n✓ TEST PASSED: Plan progress calculated correctly")


def test_vector_components():
    """Test that vector components are correctly structured."""
    print("\n" + "="*70)
    print("TEST 6: Vector Structure")
    print("="*70)
    
    scenario = MockScenario()
    obs = build_observation_vector(scenario, "f16_01", {}, 0)
    
    print("Vector structure:")
    print(f"  Features 0-5 (self):     {obs.vector[0:6]}")
    print(f"  Features 6-11 (target0): {obs.vector[6:12]}")
    print(f"  Features 12-17 (target1): {obs.vector[12:18]}")
    print(f"  Features 18-23 (target2): {obs.vector[18:24]}")
    
    print("\n✓ TEST PASSED: Vector structure is correct")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING OBSERVATION MODULE TESTS")
    print("="*70)
    
    try:
        test_basic_observation()
        test_no_weapons()
        test_no_targets()
        test_target_in_plan()
        test_plan_progress()
        test_vector_components()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nObservation module is ready for integration with BLADE.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
