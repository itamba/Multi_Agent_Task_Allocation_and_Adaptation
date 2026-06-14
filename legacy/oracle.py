"""
Oracle Wrapper - MATCH-AOU Integration
=======================================

Wraps MATCH-AOU solver to provide "oracle" actions for imitation learning.

The oracle represents optimal decision-making:
- Knows full scenario (all targets)
- Solves optimal allocation
- Provides ground truth for RL training

Usage:
    from match_aou.rl.training import MatchAOUOracle
    
    oracle = MatchAOUOracle(solver_name='bonmin')
    
    # Get optimal action for current state
    action = oracle.get_action(observation, agent_id, tasks, solution)
"""

import logging
from typing import List, Optional, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MatchAOUOracle:
    """
    Oracle that uses MATCH-AOU to provide optimal actions.
    
    In training, we compare RL agent's decisions against this oracle.
    The oracle has access to full information (all targets), while
    RL agent discovers targets gradually.
    
    Attributes:
        solver_name: MATCH-AOU solver to use ('bonmin', 'ipopt', etc.)
        cache: Cache of solutions to avoid re-solving
    """
    
    def __init__(
        self,
        solver_name: str = 'bonmin',
        use_cache: bool = True
    ):
        """
        Initialize oracle.
        
        Args:
            solver_name: MATCH-AOU solver name
            use_cache: Whether to cache solutions
        """
        self.solver_name = solver_name
        self.use_cache = use_cache
        self.cache = {}
    
    def get_action(
        self,
        observation,  # ObservationOutput
        agent_id: str,
        tasks: List,  # List[Task]
        current_solution: Dict,  # Current MATCH-AOU solution
        scenario,  # BLADE Scenario
        new_targets: Optional[List] = None
    ) -> int:
        """
        Get optimal action from oracle.
        
        Process:
        1. If new targets discovered → re-run MATCH-AOU with full info
        2. Compare current plan to new optimal plan
        3. Return action that moves toward optimal plan
        
        Args:
            observation: Current observation
            agent_id: Agent ID
            tasks: All tasks (including new targets)
            current_solution: Current MATCH-AOU solution
            scenario: BLADE scenario
            new_targets: Newly discovered targets (if any)
        
        Returns:
            Optimal action index (0-4)
        """
        # For now, implement simple heuristic
        # Full implementation requires running MATCH-AOU solver
        
        # If no new targets, stick to plan (NOOP)
        if not new_targets or len(new_targets) == 0:
            return 0  # NOOP
        
        # If new unassigned target nearby, attack it
        for i, target in enumerate(observation.targets):
            if target.exists and not target.is_in_plan and target.distance_norm < 0.5:
                return i + 1  # INSERT_ATTACK(i)
        
        # Default: NOOP
        return 0
    
    def get_action_mask(
        self,
        observation,  # ObservationOutput
        scenario,  # BLADE Scenario
        agent_id: str,
        last_attack_tick: int = -9999
    ) -> np.ndarray:
        """
        Get valid action mask for current state.
        
        This uses the action validation from action space module.
        
        Args:
            observation: Current observation
            scenario: BLADE scenario
            agent_id: Agent ID
            last_attack_tick: Tick of last attack
        
        Returns:
            Boolean array [action_dim] where True = valid action
        """
        from .training_utils import get_action_mask_array
        return get_action_mask_array(observation, scenario, agent_id, last_attack_tick)
    
    def solve_full_problem(
        self,
        agents: List,
        tasks: List,
        precedence_relations: List
    ) -> Dict:
        """
        Solve full MATCH-AOU problem with all targets.
        
        This is the "oracle" solution that knows everything.
        
        Args:
            agents: List of Agent objects
            tasks: List of Task objects (including new targets)
            precedence_relations: Task dependencies
        
        Returns:
            Solution dict {agent_id: [(task_idx, step_idx, level), ...]}
        """
        # Import MATCH-AOU solver
        try:
            from match_aou.solvers import MatchAou
            
            solver = MatchAou(
                agents=agents,
                tasks=tasks,
                precedence_relations=precedence_relations,
                solver_name=self.solver_name
            )
            
            solution, results = solver.solve()
            
            if solution is None:
                logger.warning("MATCH-AOU solver returned no solution")
                return {}
            
            return solution
        
        except Exception as e:
            logger.error(f"Failed to solve MATCH-AOU: {e}")
            return {}
    
    def compare_solutions(
        self,
        current_solution: Dict,
        optimal_solution: Dict,
        agent_id: str
    ) -> int:
        """
        Compare current plan to optimal plan and suggest action.
        
        Args:
            current_solution: Current agent plan
            optimal_solution: Optimal plan from oracle
            agent_id: Agent ID
        
        Returns:
            Suggested action index
        """
        current_plan = current_solution.get(agent_id, [])
        optimal_plan = optimal_solution.get(agent_id, [])
        
        # If plans are identical, NOOP
        if current_plan == optimal_plan:
            return 0  # NOOP
        
        # Find first difference
        for i, (curr_task, opt_task) in enumerate(zip(current_plan, optimal_plan)):
            if curr_task != opt_task:
                # Plans diverge at position i
                # Suggest action to align with optimal
                
                # Extract target from optimal task
                # This is simplified - full implementation needs task parsing
                return 1  # INSERT_ATTACK(0) as placeholder
        
        # Default
        return 0  # NOOP


class SimpleOracle:
    """
    Simplified oracle for testing (rule-based).
    
    Uses heuristics instead of full MATCH-AOU solver.
    Useful for initial testing and debugging.
    
    Rules:
    1. If low fuel → RTB
    2. If unassigned target nearby → attack it
    3. Otherwise → NOOP (follow plan)
    """
    
    def __init__(self):
        """Initialize simple oracle."""
        self.last_decisions = {}
    
    def get_action(
        self,
        observation,  # ObservationOutput
        agent_id: str
    ) -> int:
        """
        Get action using simple heuristics.
        
        Args:
            observation: Current observation
            agent_id: Agent ID
        
        Returns:
            Action index (0-4)
        """
        # Rule 1: Low fuel → RTB
        if observation.self_state.fuel_norm < 0.2:
            if observation.self_state.rtb_possible > 0.5:
                return 4  # FORCE_RTB
        
        # Rule 2: Find unassigned target
        for i, target in enumerate(observation.targets):
            if not target.exists:
                continue
            
            # Prioritize unassigned, close targets
            if not target.is_in_plan and target.distance_norm < 0.5:
                # Check if have weapon
                if observation.self_state.has_weapon > 0.5:
                    return i + 1  # INSERT_ATTACK(i)
        
        # Rule 3: Default NOOP
        return 0  # NOOP
    
    def get_action_mask(
        self,
        observation,
        scenario,
        agent_id: str,
        last_attack_tick: int = -9999
    ) -> np.ndarray:
        """Get valid action mask (same as MatchAOUOracle)."""
        from .training_utils import get_action_mask_array
        return get_action_mask_array(observation, scenario, agent_id, last_attack_tick)

