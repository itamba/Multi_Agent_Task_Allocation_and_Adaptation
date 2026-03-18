"""
Observation Space Module

Extracts fixed-size feature vectors for RL decision-making.

Main API:
    build_observation_vector: Extract observation vector from BLADE scenario
    ObservationConfig: Configuration for observation extraction (includes top_k)
    ObservationOutput: Complete observation with metadata

Data Structures:
    SelfState: 6 self features (fuel, weapons, plan state)
    TargetInfo: Per-target information (distance, threat, etc.)

Feature Vector Structure:
    Total size: 6 + (K × 6) + 6, where K = config.top_k (default: 3)
    
    Components:
    - 6 self features (fuel, weapons, distance to next step, RTB possible, etc.)
    - K × 6 target features (K targets × 6 features each: exists, distance, threat, dynamic, in_plan, engaged)
    - 6 plan context features (fuel margin, unassigned targets, coordination, etc.)
    
    Examples:
    - top_k=3 (default): [30] = 6 + (3 × 6) + 6
    - top_k=5: [42] = 6 + (5 × 6) + 6
    
    Note: Plan context features (last 6) require tasks and solution to be provided.
          If not provided, these will be zeros.

Usage:
    from match_aou.rl.observation import build_observation_vector, ObservationConfig
    
    # Default configuration (top_k=3)
    config = ObservationConfig(top_k=3)
    obs = build_observation_vector(
        scenario=blade_scenario,
        agent_id="f16_01",
        current_plan=execution_time_to_actions,
        current_time=100,
        config=config,
        tasks=tasks,        # Optional: enables plan context features
        solution=solution   # Optional: enables plan context features
    )
    
    # obs.vector is np.array with values in [0, 1]
    print(obs.vector.shape)  # (30,) for top_k=3
    print(obs.self_state.fuel_norm)  # 0.75
    print(len(obs.targets))  # 3 (always equals top_k, padded if needed)
"""

from .config import ObservationConfig
from .observation_types  import SelfState, TargetInfo, ObservationOutput
from .observation_builder import build_observation_vector

__all__ = [
    "build_observation_vector",
    "ObservationConfig",
    "ObservationOutput",
    "SelfState",
    "TargetInfo",
]
