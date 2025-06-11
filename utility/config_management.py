import json
import itertools
from copy import deepcopy


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