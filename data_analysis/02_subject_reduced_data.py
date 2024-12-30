# To write. Will output:
# Header
#     User ID
#     Subject group (control, model card, in situ)
#     Run order
#     Demographics: Age bracket, gender, gamer status, AI familiarity, trust in automated systems
#
# Average team performance across the subject's 4 rounds:
#     Total number of targets identified across the subject's four rounds
#     Total number of weapons (threat types) identified across the subject's four rounds
#     Avg final human health at round end
#     Avg final agent health at round end
#     Avg time remaining at round end
#
# Detailed game data (totals across all 4 rounds)
#     Total mouse clicks
#     Total  human waypoints set
#     Total  waypoint override commands
#     Total  hold commands
#     Total  search area commands
#     Total  search type commands

# SA scores
    # SA level 1 score (percent of level 1 Qs correct)
    # SA level 2 score (percent of level 1 Qs correct)
    # SA level 3 score (percent of level 1 Qs correct)

# Final survey data
    # Trust score
    # Workload score
    # Weighted workload
    # Agent understanding score (self assessed)
    # Agent vignette score (percent correct)

# Plots
    # Four plots, one per round, showing time history of score and gameplan commands

import json
import csv
from pathlib import Path


# TODO
def sagat_grader():
    pass

def final_survey_summary():
    pass

def subject_plots():
    pass

# TODO check this is good
def extract_header_info(data):
    """Extract subject ID and group from header lines"""
    lines = data.split('\n')
    for line in lines:
        if '"subject_id:' in line:
            subject_id = line.split(':')[1].strip('"')
        elif '"user_group:' in line:
            user_group = line.split(':')[1].strip('"')
    return subject_id, user_group


def parse_game_file(file_path):
    """Parse a single game data file and return metrics"""
    with open(file_path, 'r') as f:
        data = f.read()

    # Extract basic info
    subject_id, user_group = extract_header_info(data)

    # Parse JSON events
    stats = {
        'targets_identified': 0,
        'weapons_identified': 0,
        'final_human_health': 0,
        'final_agent_health': 0,
        'time_remaining': 0,
        'mouse_clicks': 0,
        'human_waypoints': 0,
        'waypoint_overrides': 0,
        'hold_commands': 0,
        'search_area_commands': 0,
        'search_type_commands': 0
    }

    # Count mouse events/commands
    lines = data.split('\n')
    for line in lines:
        if not line:
            continue

        if '"type": "mouse_event"' in line:
            stats['mouse_clicks'] += 1
            event = json.loads(line)
            if 'human waypoint' in event['event_type']:
                stats['human_waypoints'] += 1

    # Get final game state metrics
    for line in reversed(lines):
        if '"type": "state"' in line:
            state = json.loads(line)
            game_state = state['game_state']
            stats['targets_identified'] = game_state['identified_targets']
            stats['weapons_identified'] = game_state['identified_threat_types']
            stats['time_remaining'] = 240 - game_state['time']  # Assuming 240s time limit
            stats['final_human_health'] = 100 - abs(game_state['agent1_damage'])
            stats['final_agent_health'] = 100 - abs(game_state['agent0_damage'])
            break

    # Parse command history
    for line in lines:
        if '"gameplan_command_history"' in line:
            commands = json.loads(line.split('"gameplan_command_history": ')[1].split(', "total_gameplan_commands"')[0])
            for cmd in commands:
                cmd_type = cmd[1]
                if cmd_type == 'waypoint':
                    stats['waypoint_overrides'] += 1
                elif cmd_type == 'hold':
                    stats['hold_commands'] += 1
                elif cmd_type in ['NW', 'NE', 'SW', 'SE', 'full']:
                    stats['search_area_commands'] += 1
                elif cmd_type in ['target_id', 'wez_id']:
                    stats['search_type_commands'] += 1

    return subject_id, user_group, stats


def process_subject_data(file_paths):
    """Process all game files for a subject and return averaged metrics"""
    total_stats = {
        'targets_identified': 0,
        'weapons_identified': 0,
        'final_human_health': 0,
        'final_agent_health': 0,
        'time_remaining': 0,
        'mouse_clicks': 0,
        'human_waypoints': 0,
        'waypoint_overrides': 0,
        'hold_commands': 0,
        'search_area_commands': 0,
        'search_type_commands': 0
    }

    # Process each file
    num_files = len(file_paths)
    for file_path in file_paths:
        subject_id, user_group, stats = parse_game_file(file_path)
        for key in total_stats:
            total_stats[key] += stats[key]

    # Calculate averages for relevant metrics
    avg_stats = {
        'subject_id': subject_id,
        'user_group': user_group,
        'avg_targets_identified': total_stats['targets_identified'] / num_files,
        'avg_weapons_identified': total_stats['weapons_identified'] / num_files,
        'avg_final_human_health': total_stats['final_human_health'] / num_files,
        'avg_final_agent_health': total_stats['final_agent_health'] / num_files,
        'avg_time_remaining': total_stats['time_remaining'] / num_files,
        'total_mouse_clicks': total_stats['mouse_clicks'],
        'total_human_waypoints': total_stats['human_waypoints'],
        'total_waypoint_overrides': total_stats['waypoint_overrides'],
        'total_hold_commands': total_stats['hold_commands'],
        'total_search_area_commands': total_stats['search_area_commands'],
        'total_search_type_commands': total_stats['search_type_commands']
    }

    return avg_stats

def write_results_csv(results, output_file):
    """Write results to CSV file"""
    fieldnames = [
        'subject_id', 'user_group',
        'avg_targets_identified', 'avg_weapons_identified',
        'avg_final_human_health', 'avg_final_agent_health',
        'avg_time_remaining', 'total_mouse_clicks',
        'total_human_waypoints', 'total_waypoint_overrides',
        'total_hold_commands', 'total_search_area_commands',
        'total_search_type_commands'
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)


def main():
    # Example usage:
    subject_files = [
        'game_data_round1.jsonl',
        'game_data_round2.jsonl',
        'game_data_round3.jsonl',
        'game_data_round4.jsonl'
    ]

    results = process_subject_data(subject_files)
    write_results_csv(results, 'subject_metrics.csv')

if __name__ == "__main__":
    main()