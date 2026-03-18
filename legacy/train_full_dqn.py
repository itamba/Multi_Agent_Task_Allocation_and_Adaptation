"""
Full Training Integration Script
=================================

Complete end-to-end training with:
- Scenario loading
- Task extraction
- MATCH-AOU solving
- Aircraft launch
- RL training
- Evaluation

Usage:
    python train_full.py --scenario strike_training_2v3.json --episodes 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from match_aou.rl.training import MatchAOUOracle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_tasks_from_scenario(scenario_path: str):
    """Extract tasks from scenario JSON."""
    with open(scenario_path, 'r') as f:
        scenario_data = json.load(f)
    
    scenario = scenario_data.get('currentScenario', {})
    
    # Get BLUE side
    blue_side_id = None
    for side in scenario.get('sides', []):
        if side['name'] == 'BLUE':
            blue_side_id = side['id']
            break
    
    if not blue_side_id:
        return []
    
    # Get hostile sides
    hostiles = scenario.get('relationships', {}).get('hostiles', {})
    hostile_sides = hostiles.get(blue_side_id, [])
    
    # Extract tasks
    tasks = []
    task_id = 0
    
    # Facilities
    for facility in scenario.get('facilities', []):
        if facility['sideId'] in hostile_sides:
            tasks.append({
                'id': task_id,
                'type': 'strike_facility',
                'target_id': facility['id'],
                'target_name': facility['name'],
                'location': {
                    'latitude': facility['latitude'],
                    'longitude': facility['longitude']
                }
            })
            task_id += 1
    
    # Airbases
    for airbase in scenario.get('airbases', []):
        if airbase['sideId'] in hostile_sides:
            tasks.append({
                'id': task_id,
                'type': 'strike_airbase',
                'target_id': airbase['id'],
                'target_name': airbase['name'],
                'location': {
                    'latitude': airbase['latitude'],
                    'longitude': airbase['longitude']
                }
            })
            task_id += 1
    
    return tasks


def extract_agents_from_scenario(scenario_path: str):
    """Extract BLUE agents from scenario."""
    with open(scenario_path, 'r') as f:
        scenario_data = json.load(f)
    
    scenario = scenario_data.get('currentScenario', {})
    
    # Get BLUE side
    blue_side_id = None
    for side in scenario.get('sides', []):
        if side['name'] == 'BLUE':
            blue_side_id = side['id']
            break
    
    agents = []
    
    # Aircraft from main list
    for ac in scenario.get('aircraft', []):
        if ac['sideId'] == blue_side_id:
            agents.append(ac)
    
    # Aircraft in airbases
    for airbase in scenario.get('airbases', []):
        if airbase['sideId'] == blue_side_id:
            for ac in airbase.get('aircraft', []):
                if ac['sideId'] == blue_side_id:
                    agents.append(ac)
    
    return agents


def train_episode_simple(
    trainer,
    oracle,
    initializer,
    scenario_path,
    max_steps=100
):
    """
    Train single episode (simplified version).
    
    Returns:
        Episode metrics
    """
    # Extract tasks and agents
    all_tasks = extract_tasks_from_scenario(scenario_path)
    agents = extract_agents_from_scenario(scenario_path)
    
    logger.info(f"\nEpisode Start: {len(agents)} agents, {len(all_tasks)} tasks")
    
    # Initialize episode
    observations, partial_solution, full_solution = initializer.initialize_episode(
        scenario=None,  # Would be BLADE scenario
        agents=agents,
        all_tasks=all_tasks,
        partial_ratio=0.67
    )
    
    # Episode statistics
    episode_reward = 0.0
    episode_steps = 0
    matches = 0
    total_decisions = 0
    
    # Simplified training loop (single agent for demo)
    for agent_id, obs in observations.items():
        for step in range(max_steps):
            # Get action mask (all valid for demo)
            action_mask = np.ones(5, dtype=np.float32)
            
            # RL selects action
            rl_action = trainer.get_action(
                state=obs.vector,
                action_mask=action_mask,
                training=True
            )
            
            # Oracle selects action (simple for demo)
            oracle_action = oracle.get_action(obs, agent_id)
            
            # Check if match
            is_match = (rl_action == oracle_action)
            if is_match:
                matches += 1
            total_decisions += 1
            
            # Compute reward
            from match_aou.rl.reward import compute_reward
            reward = compute_reward(
                rl_action=rl_action,
                oracle_action=oracle_action,
                observation=obs,
                is_valid=True
            )
            
            episode_reward += reward
            
            # Create mock next observation
            next_obs = obs  # Simplified
            done = (step >= max_steps - 1)
            
            # Store experience
            trainer.add_experience(
                state=obs.vector,
                action=rl_action,
                reward=reward,
                next_state=next_obs.vector,
                done=done,
                action_mask=action_mask,
                next_action_mask=action_mask
            )
            
            # Train
            loss = trainer.train_step()
            
            episode_steps += 1
            
            if done:
                break
    
    # Update epsilon
    trainer.update_epsilon()
    
    # Compute accuracy
    accuracy = matches / total_decisions if total_decisions > 0 else 0.0
    
    return {
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'imitation_accuracy': accuracy,
        'epsilon': trainer.epsilon
    }


def main():
    parser = argparse.ArgumentParser(description="Full RL Training")
    parser.add_argument("--scenario", default="data/scenarios/strike_training_2v3.json", help="Scenario JSON file")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--save-freq", type=int, default=10, help="Save every N episodes")
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Full Training Script")
    logger.info("="*70)
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Episodes: {args.episodes}")
    
    # Import components
    try:
        from match_aou.rl.agent import EnhancedMLPQNetwork
        from match_aou.rl.training import DQNTrainer, TrainingConfig, SimpleOracle
        from match_aou.rl.training.episode_initializer import EpisodeInitializer
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you're running from project root")
        return
    
    # Create components
    logger.info("\nInitializing components...")
    
    # Network
    q_network = EnhancedMLPQNetwork(obs_dim=30, action_dim=5)
    logger.info("✅ Q-Network created")
    
    # Trainer
    config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        buffer_size=10000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    trainer = DQNTrainer(q_network, config)
    logger.info("✅ Trainer created")
    
    # Oracle (simple for demo)
    oracle = MatchAOUOracle()
    logger.info("✅ Oracle created")
    
    # Mock BLADE env
    class MockBlade:
        def handle_aircraft_launch(self, agent_id):
            pass
        def step(self):
            pass
    
    blade = MockBlade()
    
    # Episode initializer
    initializer = EpisodeInitializer(blade, oracle)
    logger.info("✅ Initializer created")
    
    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Starting Training")
    logger.info("="*70)
    
    for episode in range(args.episodes):
        # Train episode
        metrics = train_episode_simple(
            trainer=trainer,
            oracle=oracle,
            initializer=initializer,
            scenario_path=args.scenario,
            max_steps=20  # Short episodes for demo
        )
        
        # Log
        if episode % 5 == 0:
            logger.info(
                f"Episode {episode:3d} | "
                f"Reward: {metrics['episode_reward']:6.2f} | "
                f"Accuracy: {metrics['imitation_accuracy']:5.1%} | "
                f"ε: {metrics['epsilon']:.3f}"
            )
        
        # Save checkpoint
        if episode > 0 and episode % args.save_freq == 0:
            trainer.save_checkpoint(f"checkpoint_ep{episode}.pt")
            logger.info(f"  → Saved checkpoint")
    
    # Final save
    logger.info("\n" + "="*70)
    logger.info("Training Complete!")
    logger.info("="*70)
    
    trainer.save_checkpoint("final_model.pt")
    q_network.save("models/q_network_final.pt")
    
    # Print summary
    summary = trainer.get_metrics_summary()
    logger.info("\nFinal Metrics:")
    logger.info(f"  Total steps: {summary['step_count']}")
    logger.info(f"  Total episodes: {summary['episode_count']}")
    logger.info(f"  Final ε: {summary['epsilon']:.3f}")
    logger.info(f"  Avg loss: {summary['avg_loss']:.4f}")
    logger.info(f"  Imitation accuracy: {summary['reward_stats']['accuracy']:.1%}")


if __name__ == "__main__":
    main()
