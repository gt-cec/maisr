import json
import itertools
from copy import deepcopy
from pathlib import Path


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

def load_env_config_with_sweeps(config_filename):
    """
    Load config and return a list of configs for each combination of sweep parameters.

    Args:
        config_filename: Path to JSON config file

    Returns:
        List of config dictionaries, one for each sweep combination
    """

    # Define which parameters should NOT be treated as sweep parameters (known list parameters)
    KNOWN_LIST_PARAMS = {
        'window_size',
        'agent_start_location',
        'human_start_location',
    }

    with open(config_filename, 'r') as f:
        base_config = json.load(f)

    # Identify sweep parameters (lists that aren't in KNOWN_LIST_PARAMS)
    sweep_params = {}
    fixed_params = {}

    for key, value in base_config.items():
        if isinstance(value, list) and key not in KNOWN_LIST_PARAMS:
            sweep_params[key] = value
            #print(f'Added {key}={value}')
        else:
            fixed_params[key] = value

    # If no sweep parameters, return single config
    if not sweep_params:
        return [base_config]

    # Generate all combinations of sweep parameters
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())

    configs = []
    for combination in itertools.product(*param_values):
        # Create new config for this combination
        config = deepcopy(fixed_params)

        # Add the sweep parameter values
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value

        configs.append(config)

    print(f"Generated {len(configs)} configurations from sweep parameters: {param_names}")
    return configs, param_names


def generate_sweep_run_name(config, base_run_name):
    """
    Generate a unique run name that includes sweep parameter values.

    Args:
        config: Single config dictionary
        base_run_name: Base run name from generate_run_name()

    Returns:
        Modified run name with sweep parameters
    """
    # Define which parameters should NOT be treated as sweep parameters
    KNOWN_LIST_PARAMS = {
        'window_size',
        'agent_start_location',
        'human_start_location',
    }

    # Find parameters that could be sweep parameters
    sweep_components = []

    # Common parameters that might be swept
    potential_sweep_params = [
        'entropy_regularization', 'vf_coef', 'curriculum_type',
    ]

    for param in potential_sweep_params:
        if param in config:
            value = config[param]
            # Add to run name if it's a simple value
            if isinstance(value, (int, float, str)) and not isinstance(value, list):
                if isinstance(value, float):
                    sweep_components.append(f"{param}-{value}")
                else:
                    sweep_components.append(f"{param}-{value}")

    if sweep_components:
        return base_run_name + "_" + "_".join(sweep_components)
    else:
        return base_run_name