"""
Episode Initializer
===================

Handles episode initialization including:
- Task partitioning (partial/full)
- MATCH-AOU solving
- Aircraft auto-launch
- Initial observations

Usage:
    initializer = EpisodeInitializer(blade_env, oracle)
    obs, partial_sol, full_sol = initializer.initialize_episode(
        scenario, agents, tasks
    )
"""

import logging
import random
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EpisodeInitializer:
    """
    Initialize training episodes with automatic aircraft launch.
    """
    
    def __init__(self, blade_env, oracle):
        """
        Initialize.
        
        Args:
            blade_env: BLADE environment/game instance
            oracle: Oracle for solving MATCH-AOU
        """
        self.blade = blade_env
        self.oracle = oracle
    
    def initialize_episode(
        self,
        scenario,
        agents: List,
        all_tasks: List[Dict],
        partial_ratio: float = 0.67
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Initialize episode:
        1. Create partial/full task sets
        2. Solve MATCH-AOU for both
        3. Auto-launch agents with assignments
        4. Get initial observations
        
        Args:
            scenario: BLADE scenario object
            agents: List of agent objects/dicts
            all_tasks: All tasks extracted from scenario
            partial_ratio: Fraction of tasks in partial set
        
        Returns:
            (observations, partial_solution, full_solution)
            observations: {agent_id: ObservationOutput}
            partial_solution: {agent_id: [(task, step, level), ...]}
            full_solution: {agent_id: [(task, step, level), ...]} (oracle)
        """
        logger.info("="*70)
        logger.info("Initializing Episode")
        logger.info("="*70)
        
        # Step 1: Create task sets
        partial_tasks, full_tasks = self._create_task_sets(all_tasks, partial_ratio)
        
        # Step 2: Solve both with MATCH-AOU (or oracle)
        logger.info("\nSolving MATCH-AOU...")
        partial_solution = self.oracle.solve_full_problem(
            agents, partial_tasks, precedence_relations=[]
        )
        
        full_solution = self.oracle.solve_full_problem(
            agents, full_tasks, precedence_relations=[]
        )
        
        logger.info(f"  Partial solution: {sum(len(v) for v in partial_solution.values())} assignments")
        logger.info(f"  Full solution: {sum(len(v) for v in full_solution.values())} assignments")
        
        # Step 3: Auto-launch agents
        launched = self._auto_launch_agents(agents, partial_solution)
        logger.info(f"\nLaunched {launched} aircraft")
        
        # Step 4: Wait for launch to complete
        if launched > 0:
            logger.info("Waiting for takeoff...")
            self._wait_for_takeoff(ticks=10)
        
        # Step 5: Get initial observations
        logger.info("\nGetting initial observations...")
        observations = self._get_observations(agents, scenario, partial_solution)
        
        logger.info(f"Episode initialized: {len(observations)} agents ready")
        logger.info("="*70)
        
        return observations, partial_solution, full_solution
    
    def _create_task_sets(
        self,
        all_tasks: List[Dict],
        partial_ratio: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Create partial and full task sets.
        
        Args:
            all_tasks: All tasks
            partial_ratio: Fraction for partial set
        
        Returns:
            (partial_tasks, full_tasks)
        """
        if len(all_tasks) == 0:
            return [], []
        
        # Full set = all tasks
        full_tasks = all_tasks.copy()
        
        # Partial set = random subset
        num_partial = max(1, int(len(all_tasks) * partial_ratio))
        partial_tasks = random.sample(all_tasks, num_partial)
        
        # Sort by ID
        partial_tasks = sorted(partial_tasks, key=lambda t: t['id'])
        full_tasks = sorted(full_tasks, key=lambda t: t['id'])
        
        logger.info(f"\nTask sets created:")
        logger.info(f"  Partial: {len(partial_tasks)} tasks")
        logger.info(f"  Full: {len(full_tasks)} tasks")
        logger.info(f"  Hidden: {len(full_tasks) - len(partial_tasks)} tasks")
        
        return partial_tasks, full_tasks
    
    def _auto_launch_agents(
        self,
        agents: List,
        solution: Dict
    ) -> int:
        """
        Auto-launch aircraft that have assignments.
        
        Args:
            agents: List of agents
            solution: {agent_id: [assignments]}
        
        Returns:
            Number of aircraft launched
        """
        launched = 0
        
        for agent in agents:
            agent_id = agent.get('id') or agent.id
            
            # Check if agent has assignments
            assignments = solution.get(agent_id, [])
            if len(assignments) == 0:
                logger.debug(f"Skipping {agent_id}: no assignments")
                continue
            
            # Check if agent is in airbase
            if self._is_in_airbase(agent):
                # Launch
                try:
                    self.blade.handle_aircraft_launch(agent_id)
                    logger.info(f"  Launched {agent.get('name', agent_id)}")
                    launched += 1
                except Exception as e:
                    logger.warning(f"  Failed to launch {agent_id}: {e}")
        
        return launched
    
    def _is_in_airbase(self, agent) -> bool:
        """
        Check if aircraft is in airbase.
        
        Simple heuristic: altitude == 0 or very low
        """
        # Get altitude
        altitude = agent.get('altitude', 0) if isinstance(agent, dict) else getattr(agent, 'altitude', 0)
        
        # If altitude is 0 or very low, assume in base
        return altitude < 100  # feet
    
    def _wait_for_takeoff(self, ticks: int = 10):
        """
        Wait for aircraft to take off.
        
        Args:
            ticks: Number of simulation ticks to wait
        """
        for _ in range(ticks):
            try:
                self.blade.step()
            except:
                # If blade doesn't have step(), skip
                pass
    
    def _get_observations(
        self,
        agents: List,
        scenario,
        solution: Dict
    ) -> Dict:
        """
        Get initial observations for all agents.
        
        Args:
            agents: List of agents
            scenario: BLADE scenario
            solution: Solution for observations
        
        Returns:
            {agent_id: observation}
        """
        try:
            from ..observation import build_observation_vector
        except ImportError:
            # Running as standalone test
            build_observation_vector = None
        
        observations = {}
        
        for agent in agents:
            agent_id = agent.get('id') or agent.id
            
            # Get agent's plan
            current_plan = solution.get(agent_id, [])
            
            try:
                # Build observation
                # Note: This is a simplified version
                # Real version would extract from BLADE properly
                obs = self._build_mock_observation(agent, scenario, current_plan)
                observations[agent_id] = obs
                
            except Exception as e:
                logger.warning(f"Failed to get observation for {agent_id}: {e}")
        
        return observations
    
    def _build_mock_observation(self, agent, scenario, current_plan):
        """
        Build mock observation for testing.
        
        In real implementation, this would call build_observation_vector().
        """
        # Mock observation with correct shape
        class MockObs:
            def __init__(self):
                self.vector = np.random.randn(30).astype(np.float32)
                self.vector = np.clip(self.vector, 0, 1)  # Normalize
                
                class SelfState:
                    fuel_norm = 0.8
                    has_weapon = 1.0
                    dist_to_next_step_norm = 0.5
                    next_step_is_attack = 0.0
                    rtb_possible = 1.0
                    plan_progress = 0.0
                
                self.self_state = SelfState()
                self.targets = []
                self.agent_id = agent.get('id') or agent.id
                self.current_time = 0
        
        return MockObs()


