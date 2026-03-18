"""
Plan Editor - Convert RL Actions to BLADE Commands
================================================================================

This module bridges the RL agent's discrete action space with BLADE's execution
layer. It takes action tokens from the RL policy and produces BLADE action strings
that can be executed in the simulation.

Architecture:
    RL Policy → Action Token (0-4) → Plan Editor → BLADE Action String → BLADE

Action Tokens (from action module):
    0: NOOP           - Continue as planned (returns empty string)
    1: INSERT_ATTACK_0 - Attack target from observation slot 0
    2: INSERT_ATTACK_1 - Attack target from observation slot 1
    3: INSERT_ATTACK_2 - Attack target from observation slot 2
    4: FORCE_RTB      - Return to base immediately

Key Design Decisions:
    - Does NOT regenerate full plans (that's MATCH-AOU's job)
    - Produces single BLADE action strings for immediate execution
    - Uses observation data to resolve target IDs and positions
    - Handles weapon selection automatically (uses best available)
    - Respects BLADE's one-action-per-tick constraint

Usage:
    from match_aou.rl.plan_editor import plan_edit_to_blade_action
    
    # After RL policy selects action
    action_token = 1  # INSERT_ATTACK_0
    
    # Convert to BLADE action
    blade_action = plan_edit_to_blade_action(
        action_token=action_token,
        observation_output=obs,  # From observation_builder
        scenario=blade_scenario,
        agent_id="f16_01"
    )
    
    # Execute in BLADE
    if blade_action:
        observation, reward, done, _, _ = env.step(blade_action)
    else:
        observation, reward, done, _, _ = env.step("")  # NOOP

Module Structure:
    1. Target resolution (extract ID/location from observation)
    2. Weapon selection (find best weapon for target)
    3. Action string generation (format BLADE commands)
    4. Main API function (plan_edit_to_blade_action)
    
Dependencies:
    - action module: For ActionType enum and action classification helpers
    - observation module: For ObservationOutput data structure
    - BLADE scenario: For aircraft and weapon data
"""

from dataclasses import dataclass
from typing import Optional, Any

# Import action space definitions (single source of truth)
from .action import (
    ActionType,
    is_attack_action,
    is_rtb_action, 
    is_noop_action,
    action_index_to_target_slot
)


# =============================================================================
# Target Resolution
# =============================================================================

@dataclass
class TargetData:
    """Resolved target information from observation."""
    target_id: str
    latitude: float
    longitude: float
    is_valid: bool
    
    @staticmethod
    def from_observation(observation_output, slot: int):
        """Extract target data from observation at given slot.
        
        Args:
            observation_output: ObservationOutput from observation_builder
            slot: Target slot index (0-2)
        
        Returns:
            TargetData with target info or invalid marker
        """
        if slot < 0 or slot >= len(observation_output.targets):
            return TargetData("", 0.0, 0.0, False)
        
        target_info = observation_output.targets[slot]
        
        # Check if target exists and has required fields
        if not target_info.exists:
            return TargetData("", 0.0, 0.0, False)
        
        if not target_info.id:
            return TargetData("", 0.0, 0.0, False)
        
        return TargetData(
            target_id=target_info.id,
            latitude=target_info.latitude,
            longitude=target_info.longitude,
            is_valid=True
        )


# =============================================================================
# Weapon Selection
# =============================================================================

def get_best_weapon_id(scenario: Any, agent_id: str) -> Optional[str]:
    """Get the best weapon for the agent to use.
    
    Uses BLADE's built-in weapon selection logic:
    - Prefers weapon with highest engagement range
    - Falls back to first available weapon
    - Returns None if no weapons available
    
    Args:
        scenario: BLADE Scenario object
        agent_id: Agent's unique ID
    
    Returns:
        Weapon ID string or None if no weapons
    """
    # Get aircraft from scenario
    aircraft = scenario.get_aircraft(agent_id)
    if aircraft is None:
        return None
    
    # Use BLADE's method to get best weapon
    if hasattr(aircraft, 'get_weapon_with_highest_engagement_range'):
        weapon = aircraft.get_weapon_with_highest_engagement_range()
        if weapon is not None:
            return str(weapon.id)
    
    # Fallback: get first available weapon
    weapons = getattr(aircraft, 'weapons', [])
    if weapons and len(weapons) > 0:
        return str(weapons[0].id)
    
    return None


# =============================================================================
# BLADE Action String Generation
# =============================================================================

def generate_attack_action(
    agent_id: str,
    target_data: TargetData,
    weapon_id: str,
    quantity: int = 2
) -> str:
    """Generate BLADE attack action string.
    
    Format: handle_aircraft_attack('<agent_id>', '<target_id>', '<weapon_id>', quantity)
    
    Args:
        agent_id: Attacking agent ID
        target_data: Resolved target information
        weapon_id: Weapon to use
        quantity: Number of weapons to launch (default: 2)
    
    Returns:
        BLADE action string ready for execution
    
    Example:
        >>> generate_attack_action("f16_01", target, "aim_120", 2)
        "handle_aircraft_attack('f16_01', 'sam_05', 'aim_120', 2)"
    """
    return (
        f"handle_aircraft_attack('{agent_id}', '{target_data.target_id}', "
        f"'{weapon_id}', {quantity})"
    )


def generate_rtb_action(agent_id: str) -> str:
    """Generate BLADE return-to-base action string.
    
    Format: aircraft_return_to_base('<agent_id>')
    
    Args:
        agent_id: Agent ID to send home
    
    Returns:
        BLADE action string ready for execution
    
    Example:
        >>> generate_rtb_action("f16_01")
        "aircraft_return_to_base('f16_01')"
    """
    return f"aircraft_return_to_base('{agent_id}')"


# =============================================================================
# Main API Function
# =============================================================================

def plan_edit_to_blade_action(
    action_token: int,
    observation_output,  # ObservationOutput from observation_builder
    scenario: Any,  # BLADE Scenario
    agent_id: str,
    weapon_quantity: int = 2
) -> str:
    """Convert RL action token to BLADE action string.
    
    This is the main entry point for the Plan-Edit API. It takes a discrete
    action token from the RL policy and produces a BLADE action string that
    can be executed in the simulation.
    
    Process:
        1. Decode action token (NOOP / INSERT_ATTACK_k / FORCE_RTB)
        2. For NOOP: return empty string (continue as planned)
        3. For INSERT_ATTACK_k:
           - Extract target from observation slot k
           - Find best weapon for agent
           - Generate attack action string
        4. For FORCE_RTB:
           - Generate RTB action string
    
    Args:
        action_token: Discrete action (0-4) from RL policy
        observation_output: ObservationOutput from build_observation_vector
        scenario: BLADE Scenario object (current simulation state)
        agent_id: Agent's unique ID (e.g., "f16_01")
        weapon_quantity: Number of weapons to launch (default: 2)
    
    Returns:
        BLADE action string or empty string for NOOP
        - Empty string ("") → NOOP, continue current plan
        - "handle_aircraft_attack(...)" → Execute attack
        - "aircraft_return_to_base(...)" → Execute RTB
    
    Raises:
        ValueError: If action_token is invalid or target/weapon unavailable
    
    Example:
        >>> # RL policy selected action 1 (attack target 0)
        >>> action = plan_edit_to_blade_action(
        ...     action_token=1,
        ...     observation_output=obs,
        ...     scenario=blade_scenario,
        ...     agent_id="f16_01"
        ... )
        >>> print(action)
        "handle_aircraft_attack('f16_01', 'sam_05', 'aim_120', 2)"
        
        >>> # Execute in BLADE
        >>> if action:
        ...     observation, reward, done, _, _ = env.step(action)
        ... else:
        ...     observation, reward, done, _, _ = env.step("")  # NOOP
    
    Notes:
        - NOOP returns empty string (BLADE env interprets "" as no-op)
        - Attack actions require valid target and weapon
        - RTB actions check that agent exists in scenario
        - All action strings follow BLADE's expected format
        - This function does NOT validate action legality (that's done
          in action_space validation before RL selects the action)
    """
    # Validate action token range
    if not (0 <= action_token <= 4):
        raise ValueError(f"Invalid action token: {action_token}. Must be 0-4.")
    
    # === Case 1: NOOP ===
    if is_noop_action(action_token):
        return ""  # Empty string = continue as planned
    
    # === Case 2: FORCE_RTB ===
    if is_rtb_action(action_token):
        # Verify agent exists
        aircraft = scenario.get_aircraft(agent_id)
        if aircraft is None:
            raise ValueError(f"Agent {agent_id} not found in scenario")
        
        return generate_rtb_action(agent_id)
    
    # === Case 3: INSERT_ATTACK_k ===
    if is_attack_action(action_token):
        # Get target slot using action module helper
        target_slot = action_index_to_target_slot(action_token)
        if target_slot is None:
            raise ValueError(f"Failed to extract target slot from action {action_token}")
        
        # Resolve target from observation
        target_data = TargetData.from_observation(observation_output, target_slot)
        if not target_data.is_valid:
            raise ValueError(
                f"Target slot {target_slot} is empty or invalid. "
                f"Action {action_token} cannot be executed."
            )
        
        # Get best weapon for agent
        weapon_id = get_best_weapon_id(scenario, agent_id)
        if weapon_id is None:
            raise ValueError(
                f"Agent {agent_id} has no weapons available. "
                f"Cannot execute attack action {action_token}."
            )
        
        # Generate attack action
        return generate_attack_action(
            agent_id=agent_id,
            target_data=target_data,
            weapon_id=weapon_id,
            quantity=weapon_quantity
        )
    
    # Should never reach here
    raise ValueError(f"Unhandled action token: {action_token}")


# =============================================================================
# Utility Functions
# =============================================================================

def action_token_to_string(action_token: int) -> str:
    """Convert action token to human-readable string.
    
    Useful for logging and debugging.
    
    Args:
        action_token: Action token (0-4)
    
    Returns:
        String representation (e.g., "INSERT_ATTACK(0)", "FORCE_RTB")
    
    Example:
        >>> action_token_to_string(0)
        'NOOP'
        >>> action_token_to_string(1)
        'INSERT_ATTACK(0)'
        >>> action_token_to_string(4)
        'FORCE_RTB'
    """
    if is_noop_action(action_token):
        return "NOOP"
    elif is_rtb_action(action_token):
        return "FORCE_RTB"
    elif is_attack_action(action_token):
        slot = action_index_to_target_slot(action_token)
        return f"INSERT_ATTACK({slot})"
    else:
        return f"UNKNOWN({action_token})"


def preview_blade_action(
    action_token: int,
    observation_output,
    scenario: Any,
    agent_id: str
) -> dict:
    """Preview what BLADE action will be generated without executing.
    
    Useful for debugging and validation. Returns detailed information about
    what the action would do.
    
    Args:
        action_token: Action token from RL policy
        observation_output: Observation data
        scenario: BLADE scenario
        agent_id: Agent ID
    
    Returns:
        Dictionary with:
            - action_type: String type ("NOOP", "ATTACK", "RTB")
            - blade_action: Action string that would be generated
            - target_id: Target ID (if attack)
            - weapon_id: Weapon ID (if attack)
            - valid: Whether action can be executed
            - error: Error message if invalid
    
    Example:
        >>> preview = preview_blade_action(1, obs, scenario, "f16_01")
        >>> print(preview)
        {
            'action_type': 'ATTACK',
            'blade_action': "handle_aircraft_attack('f16_01', 'sam_05', 'aim_120', 2)",
            'target_id': 'sam_05',
            'weapon_id': 'aim_120',
            'valid': True,
            'error': None
        }
    """
    result = {
        'action_type': action_token_to_string(action_token),
        'blade_action': '',
        'target_id': None,
        'weapon_id': None,
        'valid': False,
        'error': None
    }
    
    try:
        blade_action = plan_edit_to_blade_action(
            action_token=action_token,
            observation_output=observation_output,
            scenario=scenario,
            agent_id=agent_id
        )
        
        result['blade_action'] = blade_action
        result['valid'] = True
        
        # Extract additional info for attacks
        if is_attack_action(action_token):
            slot = action_index_to_target_slot(action_token)
            target_data = TargetData.from_observation(observation_output, slot)
            result['target_id'] = target_data.target_id
            result['weapon_id'] = get_best_weapon_id(scenario, agent_id)
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


# =============================================================================
# Integration Example
# =============================================================================

"""
Complete RL-to-BLADE execution flow:

1. Extract Observation:
   from match_aou.rl.observation import build_observation_vector
   
   obs = build_observation_vector(
       scenario=blade_scenario,
       agent_id="f16_01",
       current_plan=plan,
       current_time=100
   )

2. Get Valid Actions:
   from match_aou.rl.action import compute_action_mask
   
   mask = compute_action_mask(obs, blade_scenario, "f16_01")
   valid_actions = mask.get_valid_actions()

3. RL Policy Decision:
   policy_input = obs.vector  # [30] features (default top_k=3)
   action_probs = policy(policy_input)
   
   # Sample only from valid actions
   valid_probs = action_probs[valid_actions]
   valid_probs /= valid_probs.sum()
   action_token = np.random.choice(valid_actions, p=valid_probs)

4. Convert to BLADE Action (THIS MODULE):
   from match_aou.rl.plan_editor import plan_edit_to_blade_action
   
   blade_action = plan_edit_to_blade_action(
       action_token=action_token,
       observation_output=obs,
       scenario=blade_scenario,
       agent_id="f16_01"
   )

5. Execute in BLADE:
   observation, reward, done, truncated, info = env.step(blade_action)

6. Repeat:
   Go back to step 1 with new observation
"""


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'plan_edit_to_blade_action',
    'action_token_to_string',
    'preview_blade_action',
    'TargetData',
    'get_best_weapon_id',
    'generate_attack_action',
    'generate_rtb_action'
]
