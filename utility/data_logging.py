import os
import json
from datetime import datetime
import pathlib


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
                      'agent_health': env.agents[env.agent_idx].health_points,
                      'human_health': env.agents[env.human_idx].health_points,
                      'gameplan_command_history': gameplan_command_history,
                      'total_gameplan_commands': len(gameplan_command_history)}
        print('[LOGGER] Final log saved')
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
                    'identified_lowQ': float(env.low_quality_identified),
                    'identified_highQ': float(env.high_quality_identified),
                    'agent_health': env.agents[env.agent_idx].health_points,
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
                        'id': 'agent' if agent.agent_idx == env.agent_idx else 'human',
                        'position': [round(agent.x,0), round(agent.y,0)],
                        'waypoint': agent_log_info['waypoint'] if agent.agent_idx == env.agent_idx else agent1_waypoint,
                        'direction': agent.direction,
                        'priority mode': agent_log_info['priority mode'] if agent.agent_idx == env.agent_idx else 'human', # Auto or manual
                        'search type': agent_log_info['search type'] if agent.agent_idx == env.agent_idx else 'human',
                        'search area': agent_log_info['search area'] if agent.agent_idx == env.agent_idx else 'human'
                    }

                    state_data['game_state']['aircraft'].append(aircraft_data)

            print('State data: \n')
            print(state_data)
            self._write_log_entry(state_data)
            #print('Game state logged')
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