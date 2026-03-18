"""
End-to-End Test Script
======================

Tests the complete RL training pipeline:
1. Load BLADE scenario
2. Extract tasks from scenario
3. Create partial solution (oracle doesn't know all targets)
4. Create full solution (oracle knows everything)
5. Run RL training episode
6. Compare RL vs Oracle

Usage:
    python test_end_to_end.py --scenario far_scenario.json
"""

import json
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Task Extraction from Scenario
# =============================================================================

def extract_all_tasks_from_scenario(scenario_path: str) -> List[Dict]:
    """
    Extract all possible tasks from BLADE scenario.
    
    A task = attacking a hostile facility/airbase.
    
    Args:
        scenario_path: Path to scenario JSON file
    
    Returns:
        List of task dictionaries
    """
    with open(scenario_path, 'r') as f:
        scenario_data = json.load(f)
    
    scenario = scenario_data.get('currentScenario', {})
    
    # Get our side (BLUE)
    blue_side_id = None
    for side in scenario.get('sides', []):
        if side['name'] == 'BLUE':
            blue_side_id = side['id']
            break
    
    if not blue_side_id:
        logger.error("No BLUE side found in scenario")
        return []
    
    # Get hostile relationships
    hostiles = scenario.get('relationships', {}).get('hostiles', {})
    hostile_sides = hostiles.get(blue_side_id, [])
    
    # Extract all hostile targets (facilities, airbases)
    tasks = []
    task_id = 0
    
    # Facilities
    for facility in scenario.get('facilities', []):
        if facility['sideId'] in hostile_sides:
            task = {
                'id': task_id,
                'type': 'strike_facility',
                'target_id': facility['id'],
                'target_name': facility['name'],
                'target_type': 'facility',
                'location': {
                    'latitude': facility['latitude'],
                    'longitude': facility['longitude']
                },
                'priority': 1.0  # Default priority
            }
            tasks.append(task)
            task_id += 1
    
    # Airbases
    for airbase in scenario.get('airbases', []):
        if airbase['sideId'] in hostile_sides:
            task = {
                'id': task_id,
                'type': 'strike_airbase',
                'target_id': airbase['id'],
                'target_name': airbase['name'],
                'target_type': 'airbase',
                'location': {
                    'latitude': airbase['latitude'],
                    'longitude': airbase['longitude']
                },
                'priority': 1.0
            }
            tasks.append(task)
            task_id += 1
    
    logger.info(f"Extracted {len(tasks)} tasks from scenario")
    for task in tasks:
        logger.info(f"  Task {task['id']}: {task['target_name']} at ({task['location']['latitude']:.2f}, {task['location']['longitude']:.2f})")
    
    return tasks


def create_partial_and_full_task_sets(
    all_tasks: List[Dict],
    partial_ratio: float = 0.67
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create partial and full task sets.
    
    Partial: Random subset of tasks (simulates limited knowledge)
    Full: All tasks (oracle has full knowledge)
    
    Args:
        all_tasks: All extracted tasks
        partial_ratio: Fraction of tasks in partial set (0.67 = 2/3)
    
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
    
    # Sort by ID for consistency
    partial_tasks = sorted(partial_tasks, key=lambda t: t['id'])
    full_tasks = sorted(full_tasks, key=lambda t: t['id'])
    
    logger.info(f"Created task sets:")
    logger.info(f"  Partial: {len(partial_tasks)} tasks")
    logger.info(f"  Full: {len(full_tasks)} tasks")
    logger.info(f"  Hidden: {len(full_tasks) - len(partial_tasks)} tasks")
    
    return partial_tasks, full_tasks


# =============================================================================
# Mock MATCH-AOU Solver (for testing without full solver)
# =============================================================================

def mock_solve_match_aou(
    agents: List,
    tasks: List[Dict],
    scenario_data: Dict
) -> Dict:
    """
    Mock MATCH-AOU solver for testing.
    
    Creates a simple greedy assignment:
    - Agent i gets task i (round-robin)
    
    Args:
        agents: List of agent objects
        tasks: List of task dicts
        scenario_data: Scenario data
    
    Returns:
        Solution dict {agent_id: [(task_idx, step_idx, level), ...]}
    """
    if len(agents) == 0 or len(tasks) == 0:
        return {}
    
    solution = {}
    
    # Round-robin assignment
    for i, task in enumerate(tasks):
        agent_idx = i % len(agents)
        agent_id = agents[agent_idx]['id']
        
        if agent_id not in solution:
            solution[agent_id] = []
        
        # Simple: one step per task (attack)
        solution[agent_id].append((task['id'], 0, 0))
    
    logger.info("Mock MATCH-AOU solution:")
    for agent_id, assignments in solution.items():
        logger.info(f"  Agent {agent_id}: {len(assignments)} tasks")
    
    return solution


# =============================================================================
# RL vs Oracle Comparison
# =============================================================================

def compare_solutions(
    partial_solution: Dict,
    full_solution: Dict
) -> Dict:
    """
    Compare partial solution (what RL starts with) to full solution (oracle).
    
    Metrics:
    - Task overlap: How many tasks are the same?
    - Agent workload difference
    - Plan similarity
    
    Args:
        partial_solution: Solution with partial tasks
        full_solution: Solution with full tasks (oracle)
    
    Returns:
        Comparison metrics
    """
    metrics = {
        'partial_tasks': sum(len(tasks) for tasks in partial_solution.values()),
        'full_tasks': sum(len(tasks) for tasks in full_solution.values()),
        'new_tasks': 0,
        'overlap': 0.0
    }
    
    # Count new tasks
    partial_task_ids = set()
    for tasks in partial_solution.values():
        for task_idx, _, _ in tasks:
            partial_task_ids.add(task_idx)
    
    full_task_ids = set()
    for tasks in full_solution.values():
        for task_idx, _, _ in tasks:
            full_task_ids.add(task_idx)
    
    new_task_ids = full_task_ids - partial_task_ids
    metrics['new_tasks'] = len(new_task_ids)
    
    # Overlap ratio
    if len(full_task_ids) > 0:
        metrics['overlap'] = len(partial_task_ids) / len(full_task_ids)
    
    logger.info("Solution comparison:")
    logger.info(f"  Partial solution: {metrics['partial_tasks']} tasks")
    logger.info(f"  Full solution: {metrics['full_tasks']} tasks")
    logger.info(f"  New tasks discovered: {metrics['new_tasks']}")
    logger.info(f"  Overlap: {metrics['overlap']:.1%}")
    
    return metrics


# =============================================================================
# End-to-End Test
# =============================================================================

def test_end_to_end(scenario_path: str):
    """
    Complete end-to-end test.
    
    1. Load scenario
    2. Extract tasks
    3. Create partial/full sets
    4. Solve both
    5. Compare
    """
    logger.info("="*70)
    logger.info("End-to-End Test: RL Training Pipeline")
    logger.info("="*70)
    
    # Step 1: Load scenario
    logger.info("\nStep 1: Load Scenario")
    logger.info("-" * 70)
    
    with open(scenario_path, 'r') as f:
        scenario_data = json.load(f)
    
    scenario = scenario_data['currentScenario']
    logger.info(f"Scenario: {scenario['name']}")
    
    # Get agents (BLUE aircraft)
    blue_side_id = None
    for side in scenario['sides']:
        if side['name'] == 'BLUE':
            blue_side_id = side['id']
            break
    
    # Get agents (BLUE aircraft)
    blue_side_id = None
    for side in scenario['sides']:
        if side['name'] == 'BLUE':
            blue_side_id = side['id']
            break
    
    # Get aircraft from main aircraft list
    agents = [ac for ac in scenario['aircraft'] if ac['sideId'] == blue_side_id]
    
    # Also get aircraft inside airbases
    for airbase in scenario.get('airbases', []):
        if airbase['sideId'] == blue_side_id:
            for ac in airbase.get('aircraft', []):
                if ac['sideId'] == blue_side_id:
                    agents.append(ac)
    
    logger.info(f"Agents: {len(agents)}")
    for agent in agents:
        logger.info(f"  - {agent['name']}")
    
    # Step 2: Extract tasks
    logger.info("\nStep 2: Extract Tasks")
    logger.info("-" * 70)
    
    all_tasks = extract_all_tasks_from_scenario(scenario_path)
    
    if len(all_tasks) == 0:
        logger.error("No tasks found! Cannot proceed.")
        return
    
    # Step 3: Create partial/full sets
    logger.info("\nStep 3: Create Partial/Full Task Sets")
    logger.info("-" * 70)
    
    partial_tasks, full_tasks = create_partial_and_full_task_sets(all_tasks, partial_ratio=0.67)
    
    # Step 4: Solve both
    logger.info("\nStep 4: Solve MATCH-AOU")
    logger.info("-" * 70)
    
    logger.info("Solving with PARTIAL tasks (what RL starts with)...")
    partial_solution = mock_solve_match_aou(agents, partial_tasks, scenario_data)
    
    logger.info("\nSolving with FULL tasks (oracle knows everything)...")
    full_solution = mock_solve_match_aou(agents, full_tasks, scenario_data)
    
    # Step 5: Compare
    logger.info("\nStep 5: Compare Solutions")
    logger.info("-" * 70)
    
    metrics = compare_solutions(partial_solution, full_solution)
    
    # Step 6: What RL needs to learn
    logger.info("\nStep 6: RL Learning Objective")
    logger.info("-" * 70)
    logger.info("RL agent starts with PARTIAL solution")
    logger.info("During execution, agent discovers NEW targets")
    logger.info("RL must learn to:")
    logger.info(f"  1. Detect {metrics['new_tasks']} new targets")
    logger.info(f"  2. Decide: attack new targets OR stick to plan?")
    logger.info(f"  3. Match ORACLE decision (full solution)")
    logger.info("")
    logger.info("Success metric: % agreement with oracle actions")
    
    logger.info("\n" + "="*70)
    logger.info("Test Complete!")
    logger.info("="*70)
    
    return {
        'scenario': scenario_path,
        'num_agents': len(agents),
        'num_tasks': len(all_tasks),
        'partial_tasks': len(partial_tasks),
        'full_tasks': len(full_tasks),
        'new_tasks': metrics['new_tasks'],
        'partial_solution': partial_solution,
        'full_solution': full_solution,
        'metrics': metrics
    }


# =============================================================================
# Scenario Generation Helper
# =============================================================================

def suggest_training_scenario():
    """
    Suggest a good training scenario design.
    """
    print("""
    📋 Recommended Training Scenario Design
    ========================================
    
    Name: "Strike Training - 2v3"
    
    BLUE Side (Attackers):
    ----------------------
    Aircraft 1: F-16C Fighting Falcon
      - Position: (26.0, 44.0)
      - Altitude: 10,000 ft
      - Fuel: 100% (12,000 lbs)
      - Weapons:
        * 2x AIM-120 AMRAAM (air-to-air)
        * 2x AGM-65 Maverick (air-to-ground)
      - Home Base: (26.0, 43.0)
    
    Aircraft 2: F-16C Fighting Falcon
      - Position: (26.2, 44.0)
      - Same loadout as Aircraft 1
    
    RED Side (Defenders):
    ---------------------
    Target 1: Military Facility (HIGH PRIORITY)
      - Position: (25.8, 51.0)
      - Type: Command Center
      - Distance from BLUE: ~700 km (CLOSE)
    
    Target 2: Airbase (MEDIUM PRIORITY)
      - Position: (25.0, 51.5)
      - Type: Fighter Base
      - Distance: ~750 km (MEDIUM)
    
    Target 3: Facility (LOW PRIORITY)
      - Position: (24.5, 52.0)
      - Type: Radar Station
      - Distance: ~850 km (FAR)
    
    Scenario Settings:
    ------------------
    - Duration: 4 hours (14400 seconds)
    - Time Compression: 1x
    - Weather: Clear
    - Doctrine: Aircraft attack hostile facilities
    
    Training Setup:
    ---------------
    1. Partial Tasks: Only Targets 1 & 2
       → MATCH-AOU assigns: Agent1→Target1, Agent2→Target2
    
    2. Full Tasks: All 3 targets
       → MATCH-AOU assigns: Agent1→Target1, Agent2→Target2+Target3
       → OR: Agent1→Target1+Target3, Agent2→Target2
    
    3. RL Challenge:
       → Start with partial plan
       → Discover Target 3 mid-flight
       → Decide: Attack new target OR stick to original plan?
       → Oracle (full solution) shows optimal decision
    
    Expected Learning:
    ------------------
    - Episode 1-20: Random (ε=1.0) → ~33% accuracy
    - Episode 20-50: Learning → ~60% accuracy
    - Episode 50+: Converged → ~80% accuracy
    
    Why This Works:
    ---------------
    ✅ Simple enough to learn quickly (2 agents, 3 targets)
    ✅ Clear decision point (when Target 3 appears)
    ✅ Trade-offs (fuel vs coverage vs priority)
    ✅ Easy to visualize and debug
    ✅ Scales to harder scenarios later
    """)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-end RL test")
    parser.add_argument("--scenario", default="far_scenario.json", help="Scenario JSON file")
    parser.add_argument("--suggest", action="store_true", help="Show scenario design suggestions")
    
    args = parser.parse_args()
    
    if args.suggest:
        suggest_training_scenario()
    else:
        # Run test
        test_end_to_end(args.scenario)
