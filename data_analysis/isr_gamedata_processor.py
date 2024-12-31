import pandas as pd
import json

"""
NEW. Takes a subject's four ISR round .jsonl files and outputs a line in a tidy-friendly excel sheet for that subject
"""

def extract_metadata(filename):
    """Extract subject_id and user_group from the log file header."""
    with open(filename, 'r') as f:
        # Read first few lines to find metadata
        for line in f:
            if 'subject_id:' in line:
                subject_id = line.strip().split(':')[1]
            elif 'user_group:' in line:
                user_group = line.strip().split(':')[1]
            elif 'run_order' in line:
                run_order = line.strip().split(':')[1]
                break
    return {
        'subject_id': subject_id,
        'user_group': user_group,
        'run_order': run_order
    }

def process_log_file(filename):
    """Process a single JSONL log file and extract relevant metrics."""
    with open(filename, 'r') as f:
        # Read all lines and get the last line which contains final stats
        lines = f.readlines()
        final_line = json.loads(lines[-1])
        final_time = json.loads(lines[-2])["timestamp"]

        # Process all lines to count waypoints
        waypoint_count = 0
        for line in lines:
            try:
                data = json.loads(line)
                if "type" in data and data["type"] == "mouse_event" and data["event_type"] == "human waypoint":
                    waypoint_count += 1
            except json.JSONDecodeError:
                continue

        # Extract gameplan command counts
        gameplan_history = final_line["gameplan_command_history"]
        total_commands = final_line["total_gameplan_commands"]

        # Count different types of commands
        search_type_commands = sum(1 for cmd in gameplan_history if cmd[1] in ["target_id", "wez_id"])
        search_area_commands = sum(1 for cmd in gameplan_history if cmd[1] in ["full", "NW", "NE", "SW", "SE"])
        hold_commands = sum(1 for cmd in gameplan_history if cmd[1] == "hold")
        waypoint_override_commands = sum(1 for cmd in gameplan_history if cmd[1] == "waypoint")

        # Get final game state from second to last line
        final_state = None
        for line in reversed(lines):
            try:
                data = json.loads(line)
                if "type" in data and data["type"] == "state":
                    final_state = data["game_state"]
                    break
            except json.JSONDecodeError:
                continue
        output = {
            'score': round(final_line["final score"],1),
            'time remaining': 241 - final_time,
            'targets': final_state["identified_targets"],
            'weapons': final_state["identified_threat_types"],
            'human_hp': 4 - final_state["agent1_damage"]/25,  # Convert damage to health
            'agent_hp': 4 - final_state["agent0_damage"]/25,
            'human_waypoints': waypoint_count,
            'total_commands': total_commands,
            'search_type_commands': search_type_commands,
            'search_area_commands': search_area_commands,
            'hold_commands': hold_commands,
            'waypoint_override_commands': waypoint_override_commands
        }
        print(output)
        return output


def process_all_rounds(round_files):
    """Create a single row of data from all round files."""
    new_row = {}

    # Get metadata from first round file
    metadata = extract_metadata(round_files[0])
    new_row['subject_id'] = metadata['subject_id'][0:-1]
    new_row['user_group'] = metadata['user_group'][0:-1]
    new_row['run_order'] = metadata['run_order'][0:-1]

    # Process each round file
    for round_num, filename in enumerate(round_files, 1):
        metrics = process_log_file(filename)

        # Add metrics to new row with round number in column names
        new_row[f'score_round{round_num}'] = metrics['score']
        new_row[f'targets_round{round_num}'] = metrics['targets']
        new_row[f'timeremaining_round{round_num}'] = metrics['time remaining']
        new_row[f'weapons_round{round_num}'] = metrics['weapons']
        new_row[f'humanhp_round{round_num}'] = metrics['human_hp']
        new_row[f'agenthp_round{round_num}'] = metrics['agent_hp']
        new_row[f'humanwaypoints_round{round_num}'] = metrics['human_waypoints']
        new_row[f'totalcommands_round{round_num}'] = metrics['total_commands']
        new_row[f'searchtypecommands_round{round_num}'] = metrics['search_type_commands']
        new_row[f'searchareacommands_round{round_num}'] = metrics['search_area_commands']
        new_row[f'holdcommands_round{round_num}'] = metrics['hold_commands']
        new_row[f'waypointoverridecommands_round{round_num}'] = metrics['waypoint_override_commands']

    return new_row


def update_excel(log_files, excel_file):
    """Update Excel file with metrics from all log files by appending a new row."""
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        # If file doesn't exist, create empty DataFrame with correct columns
        columns = ['subject_id', 'user_group','run_order']  # Add metadata columns
        for i in range(1, 5):  # For rounds 1-4
            round_cols = [
                f'score_round{i}', f'targets_round{i}', f'weapons_round{i}',
                f'humanhp_round{i}', f'agenthp_round{i}', f'timeremaining_round{i}', f'humanwaypoints_round{i}',
                f'totalcommands_round{i}', f'searchtypecommands_round{i}',
                f'searchareacommands_round{i}', f'holdcommands_round{i}',
                f'waypointoverridecommands_round{i}'
            ]
            columns.extend(round_cols)
        df = pd.DataFrame(columns=columns)

    # Process all rounds and create new row
    new_row = process_all_rounds(log_files)

    # Append the new row to the DataFrame
    df.loc[len(df)] = new_row

    # Save the updated Excel file
    df.to_excel(excel_file, index=False)


if __name__ == "__main__":
    # List of log files to process (one per round)
    log_files = [
        "MAISR_Subj_997_round1.jsonl",
        "MAISR_Subj_997_round2.jsonl",
        "MAISR_Subj_997_round3.jsonl",
        "MAISR_Subj_997_round4.jsonl",
    ]

    excel_file = "test4_isr_data_header_example.xlsx"
    update_excel(log_files, excel_file)
    print('\nWrote user data to file:', excel_file)