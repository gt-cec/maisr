import pandas as pd
import json
import os
import re
from collections import defaultdict

"""
NEW. Takes all .jsonl files from a folder and outputs them to an excel sheet (one row per subject) for later processing in R.

Important notes: 
1. Subject must have completed all four rounds + the training round for this to run correctly. It automatically ignores round 0, but round 0 must be present in the folder or it will shift all round metrics by 1 round (e.g. it will report round 2's score as round 1).
2. When a friendly target is identified, the counter for both targets ID'd AND weapons ID'd goes up by 1, even though friendlies do not have weapons. Think of this as identifying the status of the target's weapons (for friendlies, status = non existent).
3. But when tallying how many weapons and targets the AI and human identified, friendlies are NOT counted as weapon IDs.
"""

def get_quadrant(x, y):
    """Determine which quadrant a position falls into"""
    if x < 500: return "SW" if y >= 500 else "NW"
    else: return "SE" if y >= 500 else "NE"


def calculate_quadrant_times(lines):
    """Calculate time spent in same and different quadrants"""
    same_quadrant_time = 0
    diff_quadrant_time = 0
    last_timestamp = None

    for line in lines:
        try:
            data = json.loads(line)
            if "type" in data and data["type"] == "state":
                current_timestamp = data["timestamp"]

                # Get positions from game state
                aircraft = data["game_state"]["aircraft"]
                agent_pos = aircraft[0]["position"]  # AI aircraft
                human_pos = aircraft[1]["position"]  # Human aircraft

                # Determine quadrants
                agent_quadrant = get_quadrant(agent_pos[0], agent_pos[1])
                human_quadrant = get_quadrant(human_pos[0], human_pos[1])

                # Calculate time difference if not first state
                if last_timestamp is not None:
                    time_diff = current_timestamp - last_timestamp

                    # Add time to appropriate counter
                    if agent_quadrant == human_quadrant: same_quadrant_time += time_diff
                    else: diff_quadrant_time += time_diff

                last_timestamp = current_timestamp

        except json.JSONDecodeError: continue

    return same_quadrant_time, diff_quadrant_time


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
        final_time = json.loads(lines[-1])["time"]

        same_quadrant_time, diff_quadrant_time = calculate_quadrant_times(lines)
        diff_quadrant_percentage = diff_quadrant_time / (diff_quadrant_time + same_quadrant_time) * 100

        # Process all lines to count waypoints
        waypoint_count = 0
        human_targets_identified = 0
        ai_targets_identified = 0
        human_weapons_identified = 0
        ai_weapons_identified = 0

        for line in lines:
            try:
                data = json.loads(line)
                if "type" in data and data["type"] == "mouse_event" and data["event_type"] == "human waypoint":
                    waypoint_count += 1
                if "identify_type" in data:
                    if data["identify_type"] == "target_identified":
                        if data["agent_id"] == "AI":
                            ai_targets_identified += 1
                        elif data["agent_id"] == "human":
                            human_targets_identified += 1
                    elif data["identify_type"] == "weapon_identified":
                        if data["agent_id"] == "AI":
                            ai_weapons_identified += 1
                        elif data["agent_id"] == "human":
                            human_weapons_identified += 1

            except json.JSONDecodeError:
                continue

        # Extract gameplan command counts
        gameplan_history = final_line["gameplan_command_history"]
        total_commands = final_line["total_gameplan_commands"]

        # Count different types of commands
        search_type_commands = sum(1 for cmd in gameplan_history if cmd[1] in ["target_id", "wez_id"])
        search_area_commands = sum(1 for cmd in gameplan_history if cmd[1] in ["full", "NW", "NE", "SW", "SE"])
        hold_commands = sum(1 for cmd in gameplan_history if cmd[1] == "hold")
        waypoint_override_commands = sum(1 for cmd in gameplan_history if cmd[1] == "waypoint override")
        time_remaining = 240 - final_time if final_line["identified_targets"] == 60 else 0

        # Get final game state from second to last line
        #final_state = None
        for line in reversed(lines):
            try:
                data = json.loads(line)
                if "type" in data and data["type"] == "state":
                    final_state = data["game_state"]
                    break
            except json.JSONDecodeError: continue

        output = {
            'score': round(final_line["final score"],1),
            'round duration': final_time,
            'time remaining': time_remaining,
            'targets': final_line["identified_targets"],
            'weapons': final_line["identified_threat_types"],
            'human_hp': final_line["human_health"],
            'agent_hp': final_line["agent_health"],
            'human_waypoints': waypoint_count,
            'total_commands': final_line['total_gameplan_commands'],
            'search_type_commands': search_type_commands,
            'search_area_commands': search_area_commands,
            'hold_commands': hold_commands,
            'waypoint_override_commands': waypoint_override_commands,
            'human_targets_identified': human_targets_identified,
            'ai_targets_identified': ai_targets_identified,
            'human_weapons_identified': human_weapons_identified,
            'ai_weapons_identified': ai_weapons_identified,
            'diff_quadrant_percentage': diff_quadrant_percentage
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

        new_row[f'score_round{round_num - 1}'] = metrics['score']
        new_row[f'duration_round{round_num - 1}'] = metrics['round duration']
        new_row[f'targets_round{round_num - 1}'] = metrics['targets']
        new_row[f'timeremaining_round{round_num - 1}'] = metrics['time remaining']
        new_row[f'weapons_round{round_num - 1}'] = metrics['weapons']
        new_row[f'humanhp_round{round_num - 1}'] = metrics['human_hp']
        new_row[f'agenthp_round{round_num - 1}'] = metrics['agent_hp']
        new_row[f'humanwaypoints_round{round_num - 1}'] = metrics['human_waypoints']
        new_row[f'totalcommands_round{round_num - 1}'] = metrics['total_commands']
        new_row[f'searchtypecommands_round{round_num - 1}'] = metrics['search_type_commands']
        new_row[f'searchareacommands_round{round_num - 1}'] = metrics['search_area_commands']
        new_row[f'holdcommands_round{round_num - 1}'] = metrics['hold_commands']
        new_row[f'waypointoverridecommands_round{round_num - 1}'] = metrics['waypoint_override_commands']
        new_row[f'human_targets_identified_round{round_num - 1}'] = metrics['human_targets_identified']
        new_row[f'ai_targets_identified_round{round_num - 1}'] = metrics['ai_targets_identified']
        new_row[f'human_weapons_identified_round{round_num - 1}'] = metrics['human_weapons_identified']
        new_row[f'ai_weapons_identified_round{round_num - 1}'] = metrics['ai_weapons_identified']
        new_row[f'diff_quadrant_percentage_round{round_num - 1}'] = metrics['diff_quadrant_percentage']

        #print('in process all rounds')
        #print(new_row)

    return new_row


def get_subject_files(data_folder):
    """Group files by subject ID and ensure each subject has all 4 rounds."""
    subject_files = defaultdict(list)
    pattern = re.compile(r'maisr_subject(\d{3})_round(\d).*\.jsonl')

    for filename in os.listdir(data_folder):
        match = pattern.match(filename)
        if match:
            subject_id = match.group(1)
            round_num = int(match.group(2))
            subject_files[subject_id].append((round_num, os.path.join(data_folder, filename)))

    # Filter out subjects without all 4 rounds and sort files by round number
    complete_subjects = {}
    for subject_id, files in subject_files.items():
        if len(files) == 5 or True:
            complete_subjects[subject_id] = [f[1] for f in sorted(files)]
        else:
            print(f"Warning: Subject {subject_id} has {len(files)} rounds instead of 4. Skipping.")

    return complete_subjects


def process_folder(data_folder, excel_file):
    """Process all subject data in a folder and write to Excel file."""
    # Initialize DataFrame with correct columns
    columns = ['subject_id', 'user_group', 'run_order']
    for i in range(1, 5):
        round_cols = [
            f'score_round{i}', f'duration_round{i}', f'targets_round{i}', f'weapons_round{i}', f'timeremaining_round{i}',
            f'humanhp_round{i}', f'agenthp_round{i}', f'humanwaypoints_round{i}',
            f'totalcommands_round{i}', f'searchtypecommands_round{i}',
            f'searchareacommands_round{i}', f'holdcommands_round{i}',
            f'waypointoverridecommands_round{i}',
            f'human_targets_identified_round{i}',
            f'ai_targets_identified_round{i}',
            f'human_weapons_identified_round{i}',
            f'ai_weapons_identified_round{i}',
            f'diff_quadrant_percentage_round{i}'
        ]
        columns.extend(round_cols)

    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)

    # Get all subject files grouped by subject ID
    subject_files = get_subject_files(data_folder)

    # Process each subject's data
    for subject_id, files in subject_files.items():
        print(f"\nProcessing subject {subject_id}...")
        new_row = process_all_rounds(files)
        df.loc[len(df)] = new_row
        print(f"Completed processing for subject {subject_id}")

    # Save the updated Excel file
    df.to_excel(excel_file, index=False)
    print(f'\nProcessed {len(subject_files)} subjects. Data written to: {excel_file}')


if __name__ == "__main__":
    data_folder = "jan10test"  # Folder containing all JSONL files
    excel_file = "maisr_gamedata_pilot_jan10.xlsx"
    process_folder(data_folder, excel_file)