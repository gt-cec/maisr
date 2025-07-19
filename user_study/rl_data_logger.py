import json
import os
import time
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class TimestepData:
    """Data structure for a single timestep"""
    timestep: int

    # Human agent data (agent 0)
    human_position: Tuple[float, float]
    human_observation: List[float]
    human_action: int
    human_custom_waypoint: Optional[Tuple[float, float]]

    # RL agent data (agent 1)
    agent_position: Tuple[float, float]
    agent_observation: List[float]
    agent_action: int

    # Environment state
    reward: float
    cumulative_reward: float
    terminated: bool
    truncated: bool

    # Target and threat states
    target_positions: List[Tuple[float, float]]
    target_info_levels: List[float]
    threat_positions: List[Tuple[float, float]]
    threat_identified: List[bool]

    # Game state
    targets_identified_total: int
    threats_identified_total: int
    detections: int


@dataclass
class EventData:
    """Data structure for game events"""
    timestep: int
    event_type: str  # 'target_identified', 'threat_identified', 'detection'
    event_data: Dict[str, Any]


@dataclass
class EpisodeSummary:
    """Summary data for a complete episode"""
    subject_id: int
    config: str
    agent_type: str
    level: int
    episode_start_time: str
    episode_end_time: str
    episode_duration_seconds: float

    # Performance metrics
    total_reward: float
    final_targets_identified: int
    final_threats_identified: int
    total_detections: int
    total_timesteps: int

    # Success metrics
    all_targets_identified: bool
    all_threats_identified: bool
    mission_success: bool

    # Behavioral metrics
    average_reward_per_timestep: float


class ExperimentDataLogger:
    """Comprehensive data logger for MAISR user study experiments"""

    def __init__(self, subject_id: int, output_dir: str = "userstudy_logs"):
        self.subject_id = subject_id
        self.output_dir = output_dir
        self.session_start_time = datetime.now()

        # Create output directory structure
        self._setup_directories()

        # Current episode data
        self.current_episode_data = {
            'timesteps': [],
            'events': [],
            'episode_info': {}
        }

        # Session-level data
        self.session_data = {
            'subject_id': subject_id,
            'session_start_time': self.session_start_time.isoformat(),
            'episodes': []
        }

        self.current_timestep = 0
        self.episode_start_time = None
        self.cumulative_reward = 0.0

        print(f"Data logger initialized for Subject {subject_id}")
        print(f"Output directory: {self.output_dir}")

    def _setup_directories(self):
        """Create directory structure for data storage"""
        # Main subject directory
        subject_dir = os.path.join(self.output_dir, f"subject_{self.subject_id}")
        os.makedirs(subject_dir, exist_ok=True)

        # Subdirectories
        self.timestep_dir = os.path.join(subject_dir, "timestep_data")
        self.summary_dir = os.path.join(subject_dir, "episode_summaries")
        self.session_dir = os.path.join(subject_dir, "session_data")

        for directory in [self.timestep_dir, self.summary_dir, self.session_dir]:
            os.makedirs(directory, exist_ok=True)

    def start_episode(self, config: str, agent_type: str, level: int):
        """Initialize logging for a new episode"""
        self.episode_start_time = datetime.now()
        self.current_timestep = 0
        self.cumulative_reward = 0.0

        # Reset episode data
        self.current_episode_data = {
            'timesteps': [],
            'events': [],
            'episode_info': {
                'config': config,
                'agent_type': agent_type,
                'level': level,
                'start_time': self.episode_start_time.isoformat()
            }
        }

        print(f"Started logging episode: {config} (Agent {agent_type}, Level {level})")

    def log_timestep(self, env, human_controller, human_action, agent_action,
                     reward: float, terminated: bool, truncated: bool, info: Dict):
        """Log data for a single timestep"""

        self.cumulative_reward += reward

        # Extract human agent data (agent 0)
        human_agent = env.envs[0].env.agents[env.envs[0].env.aircraft_ids[0]]
        human_position = (float(human_agent.x), float(human_agent.y))

        # Get human observation
        human_obs = env.envs[0].get_observation(agent_id=0)
        if hasattr(human_obs, 'tolist'):
            human_obs_list = human_obs.tolist()
        else:
            human_obs_list = list(human_obs)

        # Extract RL agent data (agent 1)
        agent_agent = env.envs[0].env.agents[env.envs[0].env.aircraft_ids[1]]
        agent_position = (float(agent_agent.x), float(agent_agent.y))

        # Get RL agent observation
        agent_obs = env.envs[0].get_observation(agent_id=1)
        if hasattr(agent_obs, 'tolist'):
            agent_obs_list = agent_obs.tolist()
        else:
            agent_obs_list = list(agent_obs)


        # Extract environment state
        target_positions = [(float(t[3]), float(t[4])) for t in env.envs[0].env.targets]
        target_info_levels = [float(t[2]) for t in env.envs[0].env.targets]
        threat_positions = [(float(t[0]), float(t[1])) for t in env.envs[0].env.threats]
        threat_identified = [bool(t) for t in env.envs[0].env.threat_identified]

        # Create timestep data
        timestep_data = TimestepData(
            timestep=self.current_timestep,
            human_position=human_position,
            human_observation=human_obs_list,
            human_action=human_action,
            human_custom_waypoint=tuple(
                human_controller.custom_waypoint) if human_controller.custom_waypoint is not None else None,
            agent_position=agent_position,
            agent_observation=agent_obs_list,
            agent_action=agent_action,
            reward=float(reward),
            cumulative_reward=float(self.cumulative_reward),
            terminated=terminated,
            truncated=truncated,
            target_positions=target_positions,
            target_info_levels=target_info_levels,
            threat_positions=threat_positions,
            threat_identified=threat_identified,
            targets_identified_total=int(env.envs[0].env.targets_identified),
            threats_identified_total=int(env.envs[0].env.num_threats_identified),
            detections=int(env.envs[0].env.detections)
        )

        self.current_episode_data['timesteps'].append(asdict(timestep_data))

        # Log events from info dict
        if 'new_identifications' in info:
            for identification in info['new_identifications']:
                self.log_event(
                    event_type=identification['type'],
                    event_data=identification
                )

        self.current_timestep += 1

    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log a game event"""
        event = EventData(
            timestep=self.current_timestep,
            event_type=event_type,
            event_data=event_data
        )

        self.current_episode_data['events'].append(asdict(event))
        print(f"Event logged: {event_type} at timestep {self.current_timestep}")

    def end_episode(self, env, final_info: Dict) -> EpisodeSummary:
        """Finalize episode logging and create summary"""
        episode_end_time = datetime.now()
        episode_duration = (episode_end_time - self.episode_start_time).total_seconds()


        # Create episode summary
        summary = EpisodeSummary(
            subject_id=self.subject_id,
            config=self.current_episode_data['episode_info']['config'],
            agent_type=self.current_episode_data['episode_info']['agent_type'],
            level=self.current_episode_data['episode_info']['level'],
            episode_start_time=self.episode_start_time.isoformat(),
            episode_end_time=episode_end_time.isoformat(),
            episode_duration_seconds=episode_duration,
            total_reward=float(self.cumulative_reward),
            final_targets_identified=int(env.envs[0].env.targets_identified),
            final_threats_identified=int(env.envs[0].env.num_threats_identified),
            total_detections=int(env.envs[0].env.detections),
            total_timesteps=self.current_timestep,
            all_targets_identified=bool(env.envs[0].env.all_targets_identified),
            all_threats_identified=bool(env.envs[0].env.all_threats_identified),
            mission_success=bool(env.envs[0].env.all_targets_identified and env.envs[0].env.all_threats_identified),
            average_reward_per_timestep=float(self.cumulative_reward / max(1, self.current_timestep))
        )

        # Save episode data
        self._save_episode_data(summary)

        # Add to session data
        self.session_data['episodes'].append(asdict(summary))

        print(f"Episode completed: {summary.config}")
        print(f"  Duration: {episode_duration:.2f} seconds")
        print(f"  Total reward: {summary.total_reward:.2f}")
        print(f"  Targets identified: {summary.final_targets_identified}")
        print(f"  Threats identified: {summary.final_threats_identified}")

        return summary

    def _save_episode_data(self, summary: EpisodeSummary):
        """Save episode data to files"""
        config_safe = summary.config.replace('/', '_')
        timestamp = self.episode_start_time.strftime("%Y%m%d_%H%M%S")

        def convert_ndarrays(obj, path=""):
            """Recursively convert numpy types (arrays, scalars) to JSON-serializable types"""
            if isinstance(obj, np.ndarray):
                #print(f"Found ndarray at {path}: shape={obj.shape}, dtype={obj.dtype}")
                return obj.tolist()
            elif isinstance(obj, (np.generic,)):  # Handle scalar types like np.float32, np.int64, etc.
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_ndarrays(v, f"{path}.{k}") for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_ndarrays(item, f"{path}[{i}]") for i, item in enumerate(obj)]
            elif isinstance(obj, tuple):
                return tuple(convert_ndarrays(item, f"{path}[{i}]") for i, item in enumerate(obj))
            else:
                return obj

        # Convert any numpy arrays in episode data
        episode_data_clean = convert_ndarrays(self.current_episode_data, "episode_data")

        # Save detailed timestep data
        timestep_filename = f"timesteps_{config_safe}_{timestamp}.json"
        timestep_path = os.path.join(self.timestep_dir, timestep_filename)

        try:
            with open(timestep_path, 'w') as f:
                json.dump(episode_data_clean, f, indent=2)
            print(f"  Timesteps: {timestep_path}")
        except TypeError as e:
            print(f"Error saving timestep data: {e}")
            # Additional debugging - check for remaining non-serializable objects
            self._debug_json_serialization(episode_data_clean, "episode_data")

        # Convert any numpy arrays in summary data
        summary_dict = asdict(summary)
        summary_clean = convert_ndarrays(summary_dict, "summary")

        # Save episode summary
        summary_filename = f"summary_{config_safe}_{timestamp}.json"
        summary_path = os.path.join(self.summary_dir, summary_filename)

        try:
            with open(summary_path, 'w') as f:
                json.dump(summary_clean, f, indent=2)
            print(f"  Summary: {summary_path}")
        except TypeError as e:
            print(f"Error saving summary data: {e}")
            # Additional debugging - check for remaining non-serializable objects
            self._debug_json_serialization(summary_clean, "summary")

        print(f"Episode data saved:")

    def _debug_json_serialization(self, obj, path=""):
        """Debug helper to identify non-JSON-serializable objects"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    json.dumps(v)
                except TypeError:
                    print(f"Non-serializable object at {path}.{k}: {type(v)} = {v}")
                    if hasattr(v, '__dict__'):
                        print(f"  Object attributes: {list(v.__dict__.keys())}")
                    self._debug_json_serialization(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                try:
                    json.dumps(item)
                except TypeError:
                    print(f"Non-serializable object at {path}[{i}]: {type(item)} = {item}")
                    if hasattr(item, '__dict__'):
                        print(f"  Object attributes: {list(item.__dict__.keys())}")
                    self._debug_json_serialization(item, f"{path}[{i}]")
        else:
            try:
                json.dumps(obj)
            except TypeError:
                print(f"Non-serializable object at {path}: {type(obj)} = {obj}")
                if hasattr(obj, '__dict__'):
                    print(f"  Object attributes: {list(obj.__dict__.keys())}")
                # Check if it's a numpy type
                if hasattr(obj, 'dtype'):
                    print(f"  Numpy dtype: {obj.dtype}")
                if hasattr(obj, 'shape'):
                    print(f"  Numpy shape: {obj.shape}")

    # def _save_episode_data(self, summary: EpisodeSummary):
    #     """Save episode data to files"""
    #     config_safe = summary.config.replace('/', '_')
    #     timestamp = self.episode_start_time.strftime("%Y%m%d_%H%M%S")
    #
    #     # Save detailed timestep data
    #     timestep_filename = f"timesteps_{config_safe}_{timestamp}.json"
    #     timestep_path = os.path.join(self.timestep_dir, timestep_filename)
    #
    #     with open(timestep_path, 'w') as f:
    #         json.dump(self.current_episode_data, f, indent=2)
    #
    #     # Save episode summary
    #     summary_filename = f"summary_{config_safe}_{timestamp}.json"
    #     summary_path = os.path.join(self.summary_dir, summary_filename)
    #
    #     with open(summary_path, 'w') as f:
    #         json.dump(asdict(summary), f, indent=2)
    #
    #     print(f"Episode data saved:")
    #     print(f"  Timesteps: {timestep_path}")
    #     print(f"  Summary: {summary_path}")

    def save_session_data(self):
        """Save complete session data"""
        self.session_data['session_end_time'] = datetime.now().isoformat()
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        self.session_data['session_duration_seconds'] = session_duration

        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        session_filename = f"session_subject_{self.subject_id}_{timestamp}.json"
        session_path = os.path.join(self.session_dir, session_filename)

        with open(session_path, 'w') as f:
            json.dump(self.session_data, f, indent=2)

            # Save survey responses
            if hasattr(self, 'survey_responses') and self.survey_responses:
                survey_file = os.path.join(self.output_dir, f'survey_responses_subject_{self.subject_id}.json')
                with open(survey_file, 'w') as f:
                    json.dump(self.survey_responses, f, indent=2)
                print(f"Survey responses saved to: {survey_file}")

        print(f"Session data saved: {session_path}")

        # Also save a summary CSV for quick analysis
        self._save_session_summary_csv()

    def _save_session_summary_csv(self):
        """Save a CSV summary of all episodes for quick analysis"""
        import csv

        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"session_summary_subject_{self.subject_id}_{timestamp}.csv"
        csv_path = os.path.join(self.session_dir, csv_filename)

        if self.session_data['episodes']:
            fieldnames = list(self.session_data['episodes'][0].keys())

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for episode in self.session_data['episodes']:
                    # Convert complex fields to strings for CSV
                    episode_copy = episode.copy()
                    writer.writerow(episode_copy)

            print(f"Session summary CSV saved: {csv_path}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the entire session"""
        if not self.session_data['episodes']:
            return {}

        episodes = self.session_data['episodes']

        summary = {
            'subject_id': self.subject_id,
            'total_episodes': len(episodes),
            'total_session_duration': sum(ep['episode_duration_seconds'] for ep in episodes),
            'average_episode_duration': sum(ep['episode_duration_seconds'] for ep in episodes) / len(episodes),
            'total_reward': sum(ep['total_reward'] for ep in episodes),
            'average_reward_per_episode': sum(ep['total_reward'] for ep in episodes) / len(episodes),
            'total_targets_identified': sum(ep['final_targets_identified'] for ep in episodes),
            'total_threats_identified': sum(ep['final_threats_identified'] for ep in episodes),
            'successful_missions': sum(1 for ep in episodes if ep['mission_success']),
            'success_rate': sum(1 for ep in episodes if ep['mission_success']) / len(episodes),
            'agent_performance': {}
        }

        # Calculate per-agent performance
        for agent_type in ['A', 'B']:
            agent_episodes = [ep for ep in episodes if ep['agent_type'] == agent_type]
            if agent_episodes:
                summary['agent_performance'][agent_type] = {
                    'episodes': len(agent_episodes),
                    'average_reward': sum(ep['total_reward'] for ep in agent_episodes) / len(agent_episodes),
                    'success_rate': sum(1 for ep in agent_episodes if ep['mission_success']) / len(agent_episodes),
                    'average_targets_identified': sum(ep['final_targets_identified'] for ep in agent_episodes) / len(
                        agent_episodes)
                }

        return summary

    def log_survey_data(self, survey_data: Dict) -> None:
        """Log survey response data"""
        if not hasattr(self, 'survey_responses'):
            self.survey_responses = []

        self.survey_responses.append({
            'episode_config': survey_data['episode_config'],
            'timestamp': survey_data['timestamp'],
            'mental_demand': survey_data['responses']["How MENTALLY demanding was the task?"],
            'physical_demand': survey_data['responses']["How PHYSICALLY demanding was the task?"],
            'temporal_demand': survey_data['responses']["How much TIME PRESSURE did you feel during the task?"],
            'effort': survey_data['responses']["How much EFFORT did the task take?"],
            'performance': survey_data['responses']["How would you rate your PERFORMANCE during the task?"],
            'frustration': survey_data['responses']["How much FRUSTRATION did you feel during the task?"]
        })

        print(f"Logged survey data for {survey_data['episode_config']}")

    def log_teammate_survey_data(self, survey_data: Dict) -> None:
        """Log teammate preference survey data"""
        if not hasattr(self, 'teammate_survey_responses'):
            self.teammate_survey_responses = []

        self.teammate_survey_responses.append({
            'survey_type': survey_data['survey_type'],
            'preferred_overall': survey_data['responses']['preferred_overall'],
            'performed_better': survey_data['responses']['performed_better'],
            'adapted_better': survey_data['responses']['adapted_better'],
            'timestamp': survey_data['timestamp']
        })

        print(f"Logged teammate preference survey data")

def load_episode_data(filepath: str) -> Dict[str, Any]:
    """Utility function to load episode data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_session_data(filepath: str) -> Dict[str, Any]:
    """Utility function to load session data from file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_timestep_data(timestep_data: List[Dict]) -> Dict[str, Any]:
    """Analyze timestep data for behavioral patterns"""
    if not timestep_data:
        return {}

    analysis = {
        'total_timesteps': len(timestep_data),
        'reward_progression': [step['cumulative_reward'] for step in timestep_data],
        'position_trajectories': {
            'human': [(step['human_position'][0], step['human_position'][1]) for step in timestep_data],
            'rl': [(step['agent_position'][0], step['agent_position'][1]) for step in timestep_data]
        },
        'custom_waypoint_timesteps': [i for i, step in enumerate(timestep_data) if
                                      step['human_custom_waypoint'] is not None]
    }


    return analysis