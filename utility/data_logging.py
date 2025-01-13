import os
import json
from datetime import datetime
import pathlib
from pathlib import Path


class GameLogger:
    def __init__(self,subject_id,config_name,user_group,round_number,run_order):
        # Create experiment_data directory if it doesn't exist
        pathlib.Path('../experiment_data').mkdir(parents=True, exist_ok=True)

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.filename = f'./experiment_data/maisr_subject{subject_id}_round{round_number}_{timestamp}.jsonl'

        self.config_name = config_name
        self.subject_id = subject_id
        self.user_group = user_group
        self.round_number = round_number
        self.run_order = run_order

        # Initialize last log time
        self.last_state_log_time = 0
        self.log_interval = 5000  # 5 seconds in milliseconds

    def initial_log(self):
        self._write_log_entry('game configuration:' + str(self.config_name))
        self._write_log_entry('subject_id:' + str(self.subject_id))
        self._write_log_entry('user_group:' + str(self.user_group))
        self._write_log_entry('round:' + str(self.round_number))
        self._write_log_entry('run_order:' + str(self.run_order))

    def final_log(self,gameplan_command_history,env):
        final_data = {'final score': round(env.score,1),
                      'identified_targets':env.identified_targets,
                      'time': round(env.display_time / 1000 - 5, 0),
                      'identified_threat_types': env.identified_threat_types,
                      'agent_health': env.agents[env.num_ships].health_points,
                      'human_health': env.agents[env.human_idx].health_points,
                      'gameplan_command_history': gameplan_command_history,
                      'total_gameplan_commands': len(gameplan_command_history)}
        self._write_log_entry(final_data)

    def log_state(self, env, current_time, agent1_waypoint,agent_log_info):
        """Log the current game state if 10 seconds have elapsed"""
        if self.last_state_log_time == 0 or (current_time - self.last_state_log_time >= self.log_interval):
            state_data = {
                'timestamp': round(current_time/1000 - 5,1),
                'type': 'state',
                'game_state': {
                    'score': env.score,
                    'time': round(env.display_time/1000,0),
                    'identified_targets': env.identified_targets,
                    'identified_threat_types': env.identified_threat_types,
                    'agent_health': env.agents[env.num_ships].health_points,
                    'human_health': env.agents[env.human_idx].health_points,
                    'ships': [],
                    'aircraft': []}
            }

            # Log each ship's state
            for agent in env.agents:
                if agent.agent_class == "ship":
                    ship_data = {
                        'id': agent.agent_idx,
                        'position': [agent.x, agent.y],
                        'threat': agent.threat,
                        'observed': agent.observed,
                        'observed_threat': agent.observed_threat
                    }
                    state_data['game_state']['ships'].append(ship_data)
                elif agent.agent_class == 'aircraft':
                    aircraft_data = {
                        'id': 'agent' if agent.agent_idx == env.num_ships else 'human',
                        'position': [round(agent.x,0), round(agent.y,0)],
                        'waypoint': agent_log_info['waypoint'] if agent.agent_idx == env.num_ships else agent1_waypoint,
                        'direction': agent.direction,
                        'priority mode': agent_log_info['priority mode'] if agent.agent_idx == env.num_ships else 'human', # Auto or manual
                        'search type': agent_log_info['search type'] if agent.agent_idx == env.num_ships else 'human',
                        'search area': agent_log_info['search area'] if agent.agent_idx == env.num_ships else 'human'
                    }

                    state_data['game_state']['aircraft'].append(aircraft_data)

            self._write_log_entry(state_data)
            print('Game state logged')
            self.last_state_log_time = current_time

    def log_mouse_event(self, event_pos, event_type, timestamp):
        """Log mouse click events (these are still logged immediately)"""
        event_data = {
            'timestamp': round(timestamp/1000 - 5,1),
            'type': 'mouse_event',
            'event_type': event_type,
            'position': event_pos
        }
        self._write_log_entry(event_data)

    def log_target_id(self, agent_id, event_type, target_id, timestamp):
        # Logs every time a target or weapon is ID'd and which player (human or AI) identified it
        event_data = {
            'timestamp': round(timestamp / 1000 - 5, 1),
            'identify_type': event_type,
            'agent_id': agent_id,
            'target_id': target_id
        }
        self._write_log_entry(event_data)
        print('Target id logged', event_data)

    def _write_log_entry(self, data):
        """Write a single entry to the log file"""
        with open(self.filename, 'a') as f:
            json.dump(data, f)
            f.write('\n')


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
            "gameboard size": (100, 2000),
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