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

import numpy as np
import wandb

from utility.config_management import load_env_config
from training.trajedi.trajedi_ppo import TrajeDiPPOTrainer


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
        "--config",
        type=str,
        default="configs/main_config.json",
        help="Path to main env config",
    )
    parser.add_argument(
        "--trajedi-config",
        type=str,
        default="configs/trajedi_config.json",
        help="Path to TrajeDi config",
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
    env_config = load_env_config(args.config)
    trajedi_config = load_trajedi_config(args.trajedi_config)

    # Override seed
    env_config["seed"] = args.seed

    # Force selfplay league type for TrajeDi (teammates are managed internally)
    env_config["league_type"] = "selfplay"

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
    print(f"Config: {args.config}")
    print(f"TrajeDi config: {args.trajedi_config}")
    print(f"\nTrajeDi settings:")
    for k, v in trajedi_config.items():
        print(f"  {k}: {v}")
    print(f"\nKey env settings:")
    print(f"  network_size: {env_config.get('network_size')}")
    print(f"  network_numlayers: {env_config.get('network_numlayers')}")
    print(f"  lr: {env_config.get('lr')}")
    print(f"  action_type: {env_config.get('action_type')}")
    print(f"  num_aircraft: {env_config.get('num_aircraft')}")

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
        print("  WandB initialized successfully")
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
