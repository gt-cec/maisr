import os
import json
from datetime import datetime
import pathlib


class GameLogger:
    def __init__(self,subject_id):
        # Create experiment_data directory if it doesn't exist
        pathlib.Path('./experiment_data').mkdir(parents=True, exist_ok=True)

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
        self.filename = f'./experiment_data/ISR_log_subject_{subject_id}_{timestamp}.jsonl'

        # Initialize last log time
        self.last_state_log_time = 0
        self.log_interval = 5000  # 10 seconds in milliseconds

    def log_state(self, env, current_time):
        """Log the current game state if 10 seconds have elapsed"""
        if self.last_state_log_time == 0 or (current_time - self.last_state_log_time >= self.log_interval):
            state_data = {
                'timestamp': current_time,
                'type': 'state',
                'game_state': {
                    'score': env.score,
                    'time': env.display_time,
                    'identified_targets': env.identified_targets,
                    'identified_threat_types': env.identified_threat_types,
                    'agent0_damage': env.agents[env.num_ships].damage,
                    'agent1_damage': env.agents[env.num_ships + 1].damage,
                    'ships': []
                }
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

            self._write_log_entry(state_data)
            print('Game state logged')
            self.last_state_log_time = current_time

    def log_mouse_event(self, event_pos, event_type, timestamp):
        """Log mouse click events (these are still logged immediately)"""
        event_data = {
            'timestamp': timestamp,
            'type': 'mouse_event',
            'event_type': event_type,
            'position': event_pos
        }
        self._write_log_entry(event_data)

    def _write_log_entry(self, data):
        """Write a single entry to the log file"""
        with open(self.filename, 'a') as f:
            json.dump(data, f)
            f.write('\n')