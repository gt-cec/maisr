"""
TrajeDi-PPO Training Entry Point for MAISR

Trains a population of SB3 PPO agents using the TrajeDi algorithm for
improved zero-shot coordination via trajectory diversity.

Usage:
    python -m training.trajedi.train_trajedi_ppo [--seed SEED] [--testing]
"""

import argparse
import json
import multiprocessing
import os
import socket
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb

from training.trajedi.trajedi_ppo import TrajeDiPPOTrainer


def load_env_config(json_path=None):
    """
    Load environment configuration from a JSON file if provided, otherwise use defaults.

    Args:
        json_path (str or Path, optional): Path to JSON configuration file

    Returns:
        dict: Environment configuration dictionary

    The function preserves default values for any parameters not specified in the JSON file.
    If the JSON file contains invalid values, it will log warnings and use defaults instead.
    """
    # Default configuration
    default_config = {
        "gameboard size": 700, # NOTE: UI elements currently do not scale based on this
        "window size": (1600,850), # width,height
        "gameboard border margin": 35,
        "gameplay color": "white",
        "motion iteration": "F",
        "search pattern": "ladder",
        "seed": 0,

        "num aircraft": 2,  # NOTE: Only two aircraft supported for now
        "num ships":30,
        "verbose": False,
        'infinite health':False,
        'time limit':120,
        'game speed':0.2, # Sets aircraft speed. 0.2 selected to set appropriate game pace: Human should have time to think about their interactions with the agent, and it should be very difficult to finish the game without the agent's help

        # Variables for situational-awareness based agent transparency study
        'show agent waypoint': 1, # Number of next waypoints to show (currently only 1 is supported)
        'show agent location': 'persistent',  # 'persistent', 'spotty', 'none' (Not implemented yet)
        'show_low_level_goals': True,
        'show_high_level_goals': True,
        'show_high_level_rationale': True,
        'show_tracked_factors': True
    }

    if json_path is None:
        return default_config

    try:
        # Convert string path to Path object if needed
        json_path = Path(json_path) if isinstance(json_path, str) else json_path

        # Check if file exists
        if not json_path.exists():
            print(f"Warning: Config file {json_path} not found. Using default configuration.")
            return default_config

        # Load JSON file
        with open(json_path, 'r') as f:
            loaded_config = json.load(f)

        # Validate and convert specific values
        if "window size" in loaded_config:
            try:
                loaded_config["window size"] = tuple(loaded_config["window size"])
            except (TypeError, ValueError):
                print("Warning: Invalid window size in config file. Using default (1600, 850)")
                loaded_config["window size"] = default_config["window size"]

        # Validate targets iteration
        if "targets iteration" in loaded_config:
            if loaded_config["targets iteration"] not in ["A", "B", "C", "D", "E"]:
                print(f"Warning: Invalid targets iteration '{loaded_config['targets iteration']}'. Using default 'C'")
                loaded_config["targets iteration"] = default_config["targets iteration"]

        # Validate show agent location
        if "show agent location" in loaded_config:
            valid_locations = ["persistent", "spotty", "none"]
            if loaded_config["show agent location"] not in valid_locations:
                print(f"Warning: Invalid show agent location value. Using default 'persistent'")
                loaded_config["show agent location"] = default_config["show agent location"]

        # Validate numeric ranges
        numeric_ranges = {
            "gameboard size": (10, 2000),
            "num aircraft": (1, 2),
            "gameboard border margin": (10, 100),
            "show agent waypoint": (0, 3),
            "time limit": (1, 600)
            #"game speed": (0.1, 10)
        }

        for key, (min_val, max_val) in numeric_ranges.items():
            if key in loaded_config:
                if not isinstance(loaded_config[key], (int, float)) or \
                        loaded_config[key] < min_val or loaded_config[key] > max_val:
                    print(f"Warning: Invalid {key} value. Using default {default_config[key]}")
                    loaded_config[key] = default_config[key]

        # Merge loaded config with defaults
        final_config = default_config.copy()
        final_config.update(loaded_config)

        return final_config

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}. Using default configuration.")
        return default_config
    except Exception as e:
        print(f"Error loading configuration: {str(e)}. Using default configuration.")
        return default_config

def load_trajedi_config(config_path: str = "configs/trajedi_config.json") -> dict:
    """Load TrajeDi-specific configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="MAISR TrajeDi-PPO Training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Testing mode with reduced parameters",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="maisr-trajedi",
        help="WandB project name",
    )
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"{'#':>5} MAISR TrajeDi-PPO Training {'#':>25}")
    print(f"{'#'*60}\n")

    # Load configs
    env_config = load_env_config("configs/main_config.json")
    trajedi_config = load_trajedi_config("configs/trajedi_config.json")

    env_config["seed"] = args.seed

    # Force selfplay league type for TrajeDi (teammates are managed internally)
    env_config["league_type"] = "selfplay"
    total_cores = multiprocessing.cpu_count()
    total_agents = trajedi_config["n_seeds"] * (trajedi_config["n_populations"] + 1)  # +1 for BR
    trajedi_config["n_envs_per_agent"] = max(1, total_cores // total_agents)

    print(f"\nParallelization:")
    print(f"  Total cores: {total_cores}")
    print(f"  Total agents: {total_agents}")
    print(f"  Envs per agent: {trajedi_config['n_envs_per_agent']}")
    print(f"  Total parallel envs: {trajedi_config['n_envs_per_agent'] * total_agents}")

    # Testing overrides
    if args.testing:
        trajedi_config["n_seeds"] = 2
        trajedi_config["n_populations"] = 2
        trajedi_config["training_rounds"] = 3
        trajedi_config["steps_per_phase"] = 2048
        trajedi_config["eval_frequency"] = 1
        trajedi_config["n_eval_episodes"] = 3
        trajedi_config["n_envs_per_agent"] = 2
        args.project = "maisr-trajedi-test"
        print("[Testing mode] Using reduced parameters")

    # Generate run name
    machine = socket.gethostname()
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = (
        f"trajedi_s{trajedi_config['n_seeds']}"
        f"p{trajedi_config['n_populations']}"
        f"_div{trajedi_config['div_factor']}"
        f"_{timestamp}_seed{args.seed}"
    )

    print(f"Run name: {run_name}")
    print(f"Machine: {machine}")

    print(f"\nTrajeDi settings:")
    for k, v in trajedi_config.items():
        if not k.startswith('_'):
            print(f"  {k}: {v}")
    print(f"\nKey env settings:")
    print(f"  network_size: {env_config.get('network_size')}")
    print(f"  network_numlayers: {env_config.get('network_numlayers')}")
    print(f"  lr: {env_config.get('lr')}")
    print(f"  action_type: {env_config.get('action_type')}")
    print(f"  num_aircraft: {env_config.get('num_aircraft')}")
    print(f'  diversity factor: {env_config.get("diversity factor")}')
    print(f'  gamma: {env_config.get("gamma")}')

    # Initialize wandb
    print(f"\nInitializing WandB (project: {args.project})...")
    wandb_run = None
    try:
        wandb_run = wandb.init(
            project=args.project,
            name=run_name + f"_{machine}",
            config={**env_config, **{"trajedi_" + k: v for k, v in trajedi_config.items()}},
            sync_tensorboard=True,
            monitor_gym=True,
        )
        wandb_run.log_code(".")

    except Exception as e:
        print(f"  WandB init failed: {e}. Continuing without logging.")

    # Create trainer and run
    trainer = TrajeDiPPOTrainer(
        env_config=env_config,
        trajedi_config=trajedi_config,
        run_name=run_name,
        project_name=args.project,
        machine_name=machine,
        wandb_run=wandb_run,
    )

    try:
        sp_results, xp_results = trainer.train()

        # Print final summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")

        print("\nSelf-play rewards (BR with itself):")
        for seed_idx, reward in sp_results.items():
            print(f"  Seed {seed_idx}: {reward:.2f}")

        print("\nCross-play rewards (BR_i with BR_j):")
        for (i, j), reward in xp_results.items():
            print(f"  Seed {i} with Seed {j}: {reward:.2f}")

        sp_mean = np.mean(list(sp_results.values()))
        xp_mean = np.mean(list(xp_results.values())) if xp_results else 0
        print(f"\nMean self-play: {sp_mean:.2f}")
        print(f"Mean cross-play: {xp_mean:.2f}")
        print(f"SP-XP gap: {sp_mean - xp_mean:.2f}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if wandb_run is not None:
            wandb_run.finish()
            print("WandB run finished.")


if __name__ == "__main__":
    main()
