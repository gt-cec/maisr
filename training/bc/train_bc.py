"""
Train a Behavior Cloning policy from expert trajectory data.

Usage:
    # Single run with defaults
    python train_bc.py expert_trajectories.pkl

    # Hyperparameter sweep
    python train_bc.py expert_trajectories.pkl --sweep

    # Custom single run
    python train_bc.py expert_trajectories.pkl --lr 1e-3 --batch_size 64 --n_epochs 50 --hidden_sizes 128 128
"""

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.algorithms.bc import BC
from imitation.data import rollout

from stable_baselines3.common.policies import ActorCriticPolicy

import json
from pathlib import Path
import torch


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

def load_trajectories(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def make_spaces(trajectories):
    """Infer observation and action spaces from trajectory data."""
    obs_dim = trajectories[0].obs.shape[1]
    n_actions = 16  # 16 discrete directions

    # Compute obs bounds from data for a tighter Box
    all_obs = np.concatenate([t.obs for t in trajectories], axis=0)
    obs_low = all_obs.min(axis=0) - 1.0
    obs_high = all_obs.max(axis=0) + 1.0

    observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
    action_space = spaces.Discrete(n_actions)
    print(f'BC training - obs dim is {obs_dim}, action space is {action_space}')
    return observation_space, action_space

def save_policy_weights_only(
    policy: ActorCriticPolicy,
    out_stem: str,
    observation_space: spaces.Space,
    action_space: spaces.Space,
    hidden_sizes: list[int],
    lr: float,
    batch_size: int,
    n_epochs: int,
    seed: int,
):
    """
    Save policy in a PyTorch-2.6-friendly format:
      - out_stem.pth  : torch.save(state_dict)  (weights-only)
      - out_stem.json : minimal JSON metadata (no gym/numpy pickling)
    """
    out_stem = str(out_stem)
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    # 1) weights only
    torch.save(policy.state_dict(), stem.with_suffix(".pth"))

    # 2) minimal metadata needed to recreate the policy module
    # Keep it pure Python types (lists, ints, floats) to avoid numpy dtype pickles.
    obs_shape = list(observation_space.shape) if getattr(observation_space, "shape", None) is not None else None

    meta = {
        "format": "bc_policy_weights_only_v1",
        "policy_class": policy.__class__.__name__,
        "obs_space": {
            "type": observation_space.__class__.__name__,
            "shape": obs_shape,
            # bounds are optional for reconstruction; include if you want exact Box recreation
            # but we can usually just rebuild from env later.
        },
        "act_space": {
            "type": action_space.__class__.__name__,
            "n": int(action_space.n) if hasattr(action_space, "n") else None,
        },
        "net_arch": [int(x) for x in hidden_sizes],
        "lr": float(lr),
        "batch_size": int(batch_size),
        "n_epochs": int(n_epochs),
        "seed": int(seed),
    }

    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))



def train_bc(
    trajectories,
    lr: float = 0.001,
    batch_size: int = 1024,
    n_epochs: int = 20,
    hidden_sizes: list[int] = [64, 64],
    seed: int = 42,
    device: str = "auto",
):
    """Train a BC policy and return the trainer + final stats."""
    rng = np.random.default_rng(seed)
    observation_space, action_space = make_spaces(trajectories)
    transitions = rollout.flatten_trajectories(trajectories)

    policy = ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: lr,
        net_arch=hidden_sizes,
    )

    trainer = BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=transitions,
        policy=policy,
        batch_size=batch_size,
        rng=rng,
        device=device,
    )

    trainer.train(n_epochs=n_epochs)

    # Evaluate: accuracy on training data
    obs = torch.tensor(transitions.obs, device=trainer.policy.device)
    true_acts = torch.tensor(transitions.acts, device=trainer.policy.device)
    with torch.no_grad():
        dist = trainer.policy.get_distribution(obs)
        pred_acts = dist.distribution.probs.argmax(dim=-1)
    accuracy = (pred_acts == true_acts).float().mean().item()

    return trainer, {"accuracy": accuracy, "loss": trainer.logger, "lr": lr, "batch_size": batch_size, "n_epochs": n_epochs, "seed": seed}


def run_sweep(trajectories, device: str = "auto"):
    """Grid search over hyperparameters."""
    param_grid = {
        "lr": [1e-3, 3e-4, 1e-4],
        "batch_size": [128, 512, 1024],
        "n_epochs": [20, 30],
        "hidden_sizes": [[64, 64]],
    }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f"Running {len(combos)} configurations...\n")

    results = []
    best_acc = 0.0
    best_config = None

    for i, values in enumerate(combos):
        config = dict(zip(keys, values))
        print(f"[{i+1}/{len(combos)}] {config}")

        trainer, stats = train_bc(trajectories, device=device, **config)

        lr, batch_size, seed, n_epochs = stats['lr'], stats['batch_size'], stats['seed'], stats['n_epochs']
        #trainer.policy.save(f"bc_policy_lr{lr}_batch{batch_size}seed{seed}epochs{n_epochs}.pt")
        out_stem = f"bc_policy_lr{lr}_batch{batch_size}seed{seed}epochs{n_epochs}"
        save_policy_weights_only(
            trainer.policy,
            out_stem,
            observation_space=trainer.observation_space,
            action_space=trainer.action_space,
            hidden_sizes=config["hidden_sizes"],
            lr=lr,
            batch_size=batch_size,
            n_epochs=n_epochs,
            seed=seed,
        )
        print(f"Saved weights-only policy to {out_stem}.pth (+ {out_stem}.json)")

        acc = stats["accuracy"]
        results.append({**config, "accuracy": acc})
        print(f"  → accuracy: {acc:.4f}\n")

        if acc > best_acc:
            best_acc = acc
            best_config = config
            best_trainer = trainer

    print("=" * 60)
    print(f"Best config: {best_config}")
    print(f"Best accuracy: {best_acc:.4f}")

    # Save best model
    #best_trainer.policy.save("bc_policy_best.pt")
    #print("Saved best policy to bc_policy_best.pt")
    save_policy_weights_only(
        best_trainer.policy,
        "bc_policy_best",
        observation_space=best_trainer.observation_space,
        action_space=best_trainer.action_space,
        hidden_sizes=best_config["hidden_sizes"],
        lr=best_config["lr"],
        batch_size=best_config["batch_size"],
        n_epochs=best_config["n_epochs"],
        seed=seed,  # same seed as used in loop; if you vary it, store per-run seed
    )
    print("Saved best weights-only policy to bc_policy_best.pth (+ bc_policy_best.json)")

    return results, best_trainer


def main():
    parser = argparse.ArgumentParser(description="Train BC from expert trajectories.")
    parser.add_argument("data", type=str, help="Path to pickled trajectories file.")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep.")
    parser.add_argument("--output", type=str, default="bc_policy.pt", help="Output model path.")
    args = parser.parse_args()

    trajectories = load_trajectories(args.data)
    print(f"Loaded {len(trajectories)} trajectories, "
          f"{sum(len(t.acts) for t in trajectories)} total state-action pairs\n")

    config_filename = 'configs/main_config.json'
    config = load_env_config(config_filename)
    config['seed'] = int(args.seed) if args.seed else 99
    config['config_filename'] = config_filename

    if args.sweep:
        run_sweep(trajectories, device=args.device)
    else:
        trainer, stats = train_bc(
            trajectories,
            lr=config['lr'],
            batch_size=config['batch_size'],
            n_epochs=args.n_epochs,
            hidden_sizes=[64, 64],
            seed=args.seed,
            device=args.device,
        )
        print(f"\nTraining accuracy: {stats['accuracy']:.4f}")
        # Treat args.output as a stem; we will write .pth and .json
        out_stem = str(Path(args.output).with_suffix(""))  # strip any .pt/.pth
        save_policy_weights_only(
            trainer.policy,
            out_stem,
            observation_space=trainer.observation_space,
            action_space=trainer.action_space,
            hidden_sizes=[64, 64],
            lr=config['lr'],
            batch_size=config['batch_size'],
            n_epochs=args.n_epochs,
            seed=args.seed,
        )
        print(f"Saved weights-only policy to {out_stem}.pth (+ {out_stem}.json)")

        #trainer.policy.save(args.output)
        #print(f"Saved policy to {args.output}")


if __name__ == "__main__":
    main()