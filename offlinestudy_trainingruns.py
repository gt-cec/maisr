#!/usr/bin/env python3
"""
Training wrapper that runs train_generic for 11 different variations of the main config.
Each variation tests different league types, seeds, network architectures, and reward configurations.
"""

import copy
import json
import argparse
import socket
import multiprocessing
from datetime import datetime
from train_generic import train_generic, load_env_config

def load_base_config(config_path="Monolith_index_August.json"):
    """Load the base configuration from JSON file."""
    return load_env_config(config_path)

def create_config_variations(base_config):
    """
    Create 11 different configuration variations based on the base config.
    
    Returns:
        List of tuples: (variation_id, description, modified_config)
    """
    variations = []
    
    # Variation 0: league_type = mixed75, seed 42
    config_0 = copy.deepcopy(base_config)
    config_0['league_type'] = 'mixed75'
    config_0['seed'] = 42
    variations.append((0, "mixed75_seed42", config_0))
    
    # Variation 1: league_type = mixed75, seed 500
    config_1 = copy.deepcopy(base_config)
    config_1['league_type'] = 'mixed75'
    config_1['seed'] = 500
    variations.append((1, "mixed75_seed500", config_1))
    
    # Variation 2: league_type = SP (selfplay), seed 42
    config_2 = copy.deepcopy(base_config)
    config_2['league_type'] = 'selfplay'
    config_2['seed'] = 42
    variations.append((2, "selfplay_seed42", config_2))

    # # Variation
    config_3 = copy.deepcopy(base_config)
    config_3['league_type'] = 'fcp'
    config_3['seed'] = 42
    variations.append((3, "fcp_seed42", config_3))

    # # Variation
    config_4 = copy.deepcopy(base_config)
    config_4['league_type'] = 'fcp'
    config_4['seed'] = 500
    variations.append((4, "fcp_seed500", config_4))
    
    # # Variation 4: league_type = strategy_diverse_nohighrisk
    # config_4 = copy.deepcopy(base_config)
    # config_4['league_type'] = 'strategy_diverse_nohighrisk'
    # variations.append((4, "strategy_diverse_nohighrisk", config_4))
    #
    # # Variation 5: league_type = strategy_diverse_nolowrisk
    # config_5 = copy.deepcopy(base_config)
    # config_5['league_type'] = 'strategy_diverse_nolowrisk'
    # variations.append((5, "strategy_diverse_nolowrisk", config_5))
    #
    # # Variation 6: league_type = strategy_diverse_nonoisy
    # config_6 = copy.deepcopy(base_config)
    # config_6['league_type'] = 'strategy_diverse_nonoisy'
    # variations.append((6, "strategy_diverse_nonoisy", config_6))
    
    # Variation 7: league_type = SP, network_size = 2x32 (reduced from default 128)
    config_5 = copy.deepcopy(base_config)
    config_5['league_type'] = 'selfplay'
    config_5['network_size'] = 32  # 2x32 architecture (pi=[32,32], vf=[32,32])
    variations.append((5, "selfplay_smallnet_2x32", config_5))
    
    # Variation 8: league_type = mixed75, network_size = 2x32
    config_6 = copy.deepcopy(base_config)
    config_6['league_type'] = 'mixed75'
    config_6['network_size'] = 32  # 2x32 architecture
    variations.append((6, "mixed75_smallnet_2x32", config_6))
    
    # Variation 9: league_type = SP, threat_id_reward = 0, threat_potential_coeff = 0
    config_7 = copy.deepcopy(base_config)
    config_7['league_type'] = 'selfplay'
    config_7['threat_id_reward'] = 0
    config_7['threat_potential_coeff'] = 0.0
    variations.append((7, "selfplay_no_threat_rewards", config_7))
    
    # Variation 10: league_type = SP, target_potential_coeff = 0, base_env_target_id_reward = 0
    config_8 = copy.deepcopy(base_config)
    config_8['league_type'] = 'selfplay'
    config_8['target_potential_coeff'] = 0.0
    config_8['base_env_target_id_reward'] = 0
    variations.append((8, "selfplay_no_target_rewards", config_8))
    
    return variations

def run_training_sweep(variations, args):
    """
    Run training for all variations or a specific subset.
    
    Args:
        variations: List of (id, description, config) tuples
        args: Command line arguments
    """
    
    # Determine machine and project settings
    machine = ('home' if socket.gethostname() == 'DESKTOP-3Q1FTUP' else 'lab' if socket.gethostname() == 'isye-ae-2023pc3' else 'pace')
    
    project_name = 'maisr-rl-configsweep'
    if args.testing:
        project_name = 'maisr-tests'
    
    num_envs = 2 if args.testing else multiprocessing.cpu_count()
    
    # Filter variations if specific ones requested
    if args.variations is not None:
        requested_ids = [int(x.strip()) for x in args.variations.split(',')]
        variations = [(vid, desc, config) for vid, desc, config in variations 
                     if vid in requested_ids]
        print(f"Running only variations: {requested_ids}")

    # Run each variation
    for variation_id, description, config in variations:
        print(f"\n{'='*80}")
        print(f"STARTING VARIATION {variation_id}: {description}")
        print(f"{'='*80}")
        
        # Modify config for testing if needed
        if args.testing:
            config["eval_freq"] = 50
            config['n_eval_episodes'] = 5
            config['save_freq'] = 500
            config['num_timesteps'] = 2000  # Reduced for testing
        
        # Set additional config parameters
        config['n_envs'] = num_envs
        config['training_variation'] = description
        
        # Generate run name with timestamp
        timestamp = datetime.now().strftime("%m%d_%H%M")
        run_name = f'configsweep_v{variation_id}_{description}_{timestamp}_seed{config["seed"]}'
        
        # Print configuration summary
        print(f"Run name: {run_name}")
        print(f"League type: {config.get('league_type', 'default')}")
        print(f"Seed: {config.get('seed', 42)}")
        print(f"Network size: {config.get('network_size', 128)}")
        print(f"Threat ID reward: {config.get('threat_id_reward', 'default')}")
        print(f"Target potential coeff: {config.get('target_potential_coeff', 'default')}")
        print(f"Environments: {num_envs}")
        print(f"Total timesteps: {config.get('num_timesteps', 'default')}")
        
        try:
            # Run training
            train_generic(
                env_config=config,
                n_envs=num_envs,
                project_name=project_name,
                use_normalize=True,
                use_teammate_manager=True,
                train_type='monolith',
                run_name=run_name,
                load_path=None,
                vecnorm_load_path=None,
                render=False,
                machine_name=machine,
                save_model=True,
                save_checkpoints=True,
                overfit_test=None,
            )
            
            print(f"✓ COMPLETED VARIATION {variation_id}: {description}")
            
        except Exception as e:
            print(f"✗ FAILED VARIATION {variation_id}: {description}")
            print(f"Error: {str(e)}")
            
            if not args.continue_on_error:
                print("Stopping due to error. Use --continue-on-error to skip failed runs.")
                raise
            else:
                print("Continuing to next variation...")
                continue
        
        print(f"{'='*80}\n")

def main():
    parser = argparse.ArgumentParser(description='Run training sweep across 11 config variations')

    parser.add_argument('--variations', type=str, default=None, help='Comma-separated list of variation IDs to run (e.g., "0,2,5"). If not specified, runs all.')
    parser.add_argument('--testing', action='store_true', help='Run in testing mode with reduced timesteps and evaluation frequency')
    parser.add_argument('--continue-on-error', action='store_true', help='Continue to next variation if one fails')
    parser.add_argument('--list-variations', action='store_true', help='List all variations and exit')

    args = parser.parse_args()
    
    # Load base configuration
    try:
        base_config = load_base_config('configs/Monolith_index_August.json')
        print(f"Loaded base configuration from: {'Monolith_index_August.json'}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create variations
    variations = create_config_variations(base_config)
    
    # List variations if requested
    if args.list_variations:
        print("\nAvailable variations:")
        print("-" * 60)
        for vid, desc, config in variations:
            league = config.get('league_type', 'default')
            seed = config.get('seed', 42)
            net_size = config.get('network_size', 128)
            print(f"{vid:2d}: {desc:25s} | League: {league:15s} | Seed: {seed:3d} | Net: {net_size}")
        print("-" * 60)
        return 0
    
    # Run the training sweep
    print(f"\nStarting training sweep for {len(variations)} variations...")
    if args.testing:
        print("TESTING MODE: Reduced timesteps and evaluation frequency")
    
    run_training_sweep(variations, args)
    
    print(f"\n🎉 Training sweep completed!")
    return 0

if __name__ == "__main__":
    main()
