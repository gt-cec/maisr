import json
import numpy as np
import datetime
import os
from typing import Dict, Any, List, Optional


class EnvironmentLogger:
    """
    Comprehensive logging system for MAISREnvVec environment
    Saves observations, actions, rewards, and environment state for debugging
    """

    def __init__(self, log_dir: str = "./logs/env_debug/", run_name: str = "default"):
        self.log_dir = log_dir
        self.run_name = run_name
        self.episode_logs = []
        self.current_episode = None

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

    def start_episode(self, episode_num: int, config: Dict[str, Any]):
        """Start logging a new episode"""

        # Debug: Check if config is actually a dict
        if not isinstance(config, dict):
            print(f"WARNING: config is not a dict, it's a {type(config)}")
            print(f"Config value: {config}")
            # Try to handle string config by parsing it as JSON
            if isinstance(config, str):
                try:
                    config = json.loads(config)
                    print("Successfully parsed config as JSON")
                except json.JSONDecodeError:
                    print("Failed to parse config as JSON, using empty dict")
                    config = {}
            else:
                config = {}

        self.current_episode = {
            "episode_number": episode_num,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": {
                "obs_type": config["obs_type"],
                "action_type": config["action_type"],
                "gameboard_size": config["gameboard_size"],
                "num_targets": config["num_targets"],
                "time_limit": config["time_limit"],
                "frame_skip": config["frame_skip"]
            },
            "steps": [],
            "episode_summary": {}
        }

    def log_step(self, step_num: int, observation: np.ndarray, action: Any,
                 reward: float, info: Dict[str, Any], env_state: Dict[str, Any]):
        """Log a single step"""
        if self.current_episode is None:
            raise ValueError("Must call start_episode() before logging steps")

        # Convert numpy arrays to lists for JSON serialization
        obs_data = observation.tolist() if isinstance(observation, np.ndarray) else observation

        # Handle different action types
        if isinstance(action, np.ndarray):
            action_data = action.tolist()
        elif isinstance(action, dict):
            action_data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in action.items()}
        else:
            action_data = action

        step_log = {
            "step": step_num,
            "observation": {
                "raw": obs_data,
                "agent_position": obs_data[:2] if len(obs_data) >= 2 else None,
                "targets_data": self._parse_target_observations(obs_data, env_state)
            },
            "action": {
                "raw": action_data,
                "processed": self._parse_action(action_data, env_state)
            },
            "reward": {
                "total": reward,
                "components": info.get("reward_breakdown", {})
            },
            "environment_state": {
                "agent_positions": env_state.get("agent_positions", []),
                "target_states": env_state.get("target_states", []),
                "detections": env_state.get("detections", 0),
                "targets_identified": env_state.get("targets_identified", 0),
                "terminated": env_state.get("terminated", False),
                "truncated": env_state.get("truncated", False)
            },
            "info": info
        }

        self.current_episode["steps"].append(step_log)

    def end_episode(self, final_reward: float, episode_length: int, success: bool):
        """Finalize episode logging"""
        if self.current_episode is None:
            return

        self.current_episode["episode_summary"] = {
            "final_reward": final_reward,
            "episode_length": episode_length,
            "success": success,
            "total_detections": sum(step["environment_state"]["detections"] for step in self.current_episode["steps"]),
            "final_targets_identified": self.current_episode["steps"][-1]["environment_state"]["targets_identified"] if
            self.current_episode["steps"] else 0
        }

        self.episode_logs.append(self.current_episode)
        self.current_episode = None

    def save_logs(self, filename_suffix: str = ""):
        """Save all episode logs to JSON file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_dir}/{self.run_name}_logs_{timestamp}{filename_suffix}.json"

        with open(filename, 'w') as f:
            json.dump({
                "run_name": self.run_name,
                "total_episodes": len(self.episode_logs),
                "episodes": self.episode_logs
            }, f, indent=2)

        print(f"Logs saved to {filename}")
        return filename

    def _parse_target_observations(self, obs: List[float], env_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse target information from observation vector"""
        if len(obs) < 2:
            return []

        targets = []
        target_start_idx = 2  # After agent x, y
        targets_per_entry = 3  # info_level, x, y
        max_targets = (len(obs) - 2) // targets_per_entry

        for i in range(max_targets):
            start_idx = target_start_idx + i * targets_per_entry
            if start_idx + 2 < len(obs):
                targets.append({
                    "target_id": i,
                    "info_level": obs[start_idx],
                    "x_position": obs[start_idx + 1],
                    "y_position": obs[start_idx + 2],
                    "is_active": obs[start_idx] > 0 or obs[start_idx + 1] > 0 or obs[start_idx + 2] > 0
                })

        return targets

    def _parse_action(self, action: Any, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """Parse action into interpretable format"""
        action_type = env_state.get("action_type", "unknown")

        if action_type == "continuous-normalized":
            return {
                "type": "continuous_waypoint",
                "normalized_x": action[0] if isinstance(action, list) and len(action) > 0 else None,
                "normalized_y": action[1] if isinstance(action, list) and len(action) > 1 else None
            }
        elif action_type == "discrete-downsampled":
            return {
                "type": "discrete_grid",
                "grid_x": action[0] if isinstance(action, list) and len(action) > 0 else None,
                "grid_y": action[1] if isinstance(action, list) and len(action) > 1 else None
            }
        elif action_type == "direct-control":
            directions = ["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"]
            return {
                "type": "direct_control",
                "direction_index": action,
                "direction_name": directions[action] if isinstance(action, int) and 0 <= action < 8 else "unknown"
            }
        else:
            return {"type": "unknown", "raw": action}


# Integration with your environment
def add_logging_to_env(env_class):
    """
    Decorator/wrapper to add comprehensive logging to your environment
    """

    class LoggedEnvironment(env_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.logger = EnvironmentLogger(
                log_dir=f"./logs/{self.tag}",
                run_name=self.run_name
            )
            self.step_count = 0
            print('Wrapped env in logger')

        def reset(self, seed=None):
            obs, info = super().reset(seed)

            # Start new episode logging
            self.logger.start_episode(
                episode_num=self.episode_counter,
                config=self.config
            )
            self.step_count = 0

            # Log initial state
            env_state = self._get_env_state()
            self.logger.log_step(
                step_num=self.step_count,
                observation=obs,
                action=None,  # No action on reset
                reward=0.0,
                info=info,
                env_state=env_state
            )

            return obs, info

        def step(self, action):
            # Call parent step method
            obs, reward, terminated, truncated, info = super().step(action)
            self.step_count += 1

            # Gather environment state
            env_state = self._get_env_state()
            env_state.update({
                "action_type": self.action_type,
                "terminated": terminated,
                "truncated": truncated
            })

            # Log this step
            self.logger.log_step(
                step_num=self.step_count,
                observation=obs,
                action=action,
                reward=reward,
                info=info,
                env_state=env_state
            )

            # End episode logging if done
            if terminated or truncated:
                self.logger.end_episode(
                    final_reward=self.ep_reward,
                    episode_length=self.step_count,
                    success=terminated and self.all_targets_identified
                )

                # Save logs periodically
                if self.episode_counter % 10 == 0:  # Save every 10 episodes
                    self.logger.save_logs(f"_ep{self.episode_counter}")

            return obs, reward, terminated, truncated, info

        def _get_env_state(self):
            """Extract current environment state for logging"""
            agent_positions = []
            for aircraft_id in self.aircraft_ids:
                if aircraft_id < len(self.agents):
                    agent = self.agents[aircraft_id]
                    agent_positions.append({
                        "agent_id": aircraft_id,
                        "x": agent.x,
                        "y": agent.y,
                        "health": agent.health_points,
                        "alive": agent.alive
                    })

            target_states = []
            for i in range(self.num_targets):
                target_states.append({
                    "target_id": int(self.targets[i, 0]),
                    "value": int(self.targets[i, 1]),  # 0=regular, 1=high-value
                    "info_level": float(self.targets[i, 2]),
                    "x": float(self.targets[i, 3]),
                    "y": float(self.targets[i, 4])
                })

            return {
                "agent_positions": agent_positions,
                "target_states": target_states,
                "detections": self.detections,
                "targets_identified": self.targets_identified,
                "episode_reward": self.ep_reward,
                "gameboard_size": self.config["gameboard_size"]
            }

    return LoggedEnvironment


# Example usage:
"""
# Wrap your environment class
LoggedMAISREnv = add_logging_to_env(MAISREnvVec)

# Create environment instance
env = LoggedMAISREnv(config=your_config, tag="debug_run")

# Use normally - logging happens automatically
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # or your policy
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

# Logs are automatically saved every 10 episodes and at the end
"""