# TODO
#  Status: Seems like every section works except the actual grading function grade_response()

"""
Claude prompt
While playing the ISR game, participants will be periodically quizzed using "SAGAT" surveys that assess their awareness of the state of the game at that instant in time. This will be done using surveys that are exported to an excel file.
Each row in the survey export excel file is a survey response. Each participant answers the survey three times per game round, for four game rounds, for a total of 12 survey responses per participant. Each question has a correct answer, and we want to grade how many questions the participant gets correct.

Each survey response has the following metadata:
* subject_id: Corresponds to the subject ID in the jsonl filenames.
* scenario_number: Which game round the survey was launched in (1,2,3,4)
* survey_number: Whether the survey was the first, second, or third survey launched for that round

This script will grade the survey responses for all participants based on whether they match the real state at the time based on the game log jsonl files, and grade each question as correct (1) or incorrect (0). 

Inputs:
* An excel file to pull SAGAT survey responses from.  
* A folder containing all .jsonl game data log files for all participants. Each participant has five jsonl files, one per game round. Round 0 is training and should be ignored.

Perform the following process for each participant:
* Loop through each row in the SAGAT survey export excel file (each row is the answers from one survey)
    - Use the subject_id variable in the survey data to identify which subject is being graded (and to know which jsonl files to use for comparison)
    - Use the "scenario number" variable in the survey metadata to access the participant's appropriate .jsonl file (scenario 1 corresponds to round1, scenario 2 corresponds to round2, and so on up to round 4)
    - Use the "survey number" variable in the survey metadata to find the correct game timestamp to use for state data comparison (survey 1 was at 60 seconds, survey 2 was at 120 sec, survey 3 was at 180 sec)
    - Grade each question, assigning "1" if correct and "0" if incorrect. Note that if the timestamps do not match up perfectly, use the game log timestamp that is closest.
        - Question 1_1 is correct if the Q1_1_x and Q1_1_y coordinates in the survey are within a 250-pixel distance of the HUMAN's true x,y pixel position for that timestamp (true position is logged in the "state" lines of the jsonl file at each timestamp
        - Question 2_1 is correct if the Q2_1_x and Q2_1_y coordinates in the survey are within a 250-pixel distance of the AGENT's true x,y pixel position for that timestamp (true position is logged in the "state" lines of the jsonl file at each timestamp
        - Q5 is correct if the answer is within 3 points of agent0's actual HEALTH at that timestamp (convert to health from damage using this formula: health = 4 - agent0_damage/25)
        - Q6 is correct if the answer is within 3 points of agent1's actual HEALTH at that timestamp (convert to health from damage using this formula: health = 4 - agent1_damage/25)
        - Q13 is correct if the answer is correct based on the percentage of targets that have been identified so far out of the total 60. (number of identified targets is listed in the jsonl state logs as identified_targets)
        - Q14 is correct if the answer matches the amount of time remaining in the round (time remaining = 245 - timestamp)
        - Q15 is correct if the answer matches the agent_mode variable in the state log for that timestamp TODO log agent's mode
        - Q11 is correct if the answer matches the final number of targets identified listed in the very last line of the log file (the line that starts with {"final score": ...)
        - Ignore questions 8, 7, and .9

Output: A new excel file with SAGAT grades, in tidyverse format. Each row corresponds to a participant's grades. The first column should be the participant ID, and all other columns will be the grade (1 or 0) for each individual question. There should be 96 grades per participant since there are 8 questions per survey for 12 surveys.
The columns should be in the following order:
| subject_id | (All question grades for round 1 survey 1) | (All grades for round 1 survey 2) | (All grades for round 1 survey 3) | (All grades for round2 survey 1) | (All grades for round2 survey 2) | and so on for all 12 surveys
"""

import pandas as pd
import json
import os
import math
from typing import Dict, List, Tuple
import glob


def load_survey_data(excel_file: str) -> pd.DataFrame:
    """Load survey responses from Excel file, skipping second header row."""
    # Read with first row as header, then skip the second row which is also header info
    df = pd.read_excel(excel_file, header=0)
    # Skip the second row (index 0 after header row)
    df = df.iloc[1:].reset_index(drop=True)
    return df


def find_jsonl_file(data_folder: str, subject_id: str, scenario_number: int) -> str:
    """Find the correct JSONL file for a given subject and scenario."""
    pattern = os.path.join(data_folder, f"maisr_subject{subject_id}_round{scenario_number}_*.jsonl")
    matching_files = glob.glob(pattern)
    return matching_files[0] if matching_files else None


# def load_game_state(jsonl_file: str, target_time: float) -> Dict:
#     """
#     Load game state from JSONL file closest to target time.
#     Returns None if file not found or no suitable state found.
#     """
#     closest_state = None
#     min_time_diff = float('inf')
#
#     try:
#         with open(jsonl_file, 'r') as f:
#             for line in f:
#                 data = json.loads(line)
#                 if 'state' in data:  # Only process state lines
#                     timestamp = data.get('timestamp', 0) / 1000  # Convert to seconds
#                     time_diff = abs(timestamp - target_time)
#
#                     if time_diff < min_time_diff:
#                         min_time_diff = time_diff
#                         closest_state = data
#
#         return closest_state
#     except (FileNotFoundError, json.JSONDecodeError):
#         return None

def load_game_state(jsonl_file: str, target_time: float) -> Dict:
    """
    Load game state from JSONL file closest to target time.
    Returns None if file not found or no suitable state found.
    """
    closest_state = None
    min_time_diff = float('inf')
    #print(f"\nSearching for game state at target time: {target_time} seconds")

    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Check if this is a state log entry containing game_state
                    if isinstance(data, dict) and data.get('type') == 'state' and 'game_state' in data:
                        # Convert time to seconds
                        timestamp = data.get('timestamp', 0)  # Already in seconds
                        time_diff = abs(timestamp - target_time)

                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_state = data['game_state']  # Store just the game_state part
                            #print(f"Found closer state at time {timestamp:.1f} sec (diff: {time_diff:.1f} sec)")

                except json.JSONDecodeError:
                    continue

        if not closest_state:
            print("No valid game state found in file")

        return closest_state
    except FileNotFoundError:
        print(f"File not found: {jsonl_file}")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None


def get_final_targets(jsonl_file: str) -> int:
    """Get final number of identified targets from last line of JSONL file."""
    try:
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()
            last_line = json.loads(lines[-1])
            return last_line.get('identified_targets', 0)
    except (FileNotFoundError, json.JSONDecodeError, IndexError):
        return 0


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    distance = math.sqrt((int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2)
    #print(distance)
    return distance


def grade_response(survey_row: pd.Series, game_state: Dict, final_targets: int) -> List[int]:
    """Grade a single survey response against the actual game state."""
    grades = []

    # Helper function to check if coordinates are within range
    def check_position(survey_x: float, survey_y: float, true_x: float, true_y: float) -> int:
        if pd.isna(survey_x) or pd.isna(survey_y):
            return 0
        distance = calculate_distance(survey_x, survey_y, true_x, true_y)
        print(f"Position check - Survey: ({survey_x}, {survey_y}), True: ({true_x}, {true_y}), Distance: {distance}")
        return 1 if distance <= 250 else 0

    # Convert damage to health
    def damage_to_health(damage: float) -> float:
        return 4 - damage / 25

    # Get actual positions and states
    state_data = game_state.get('state', {})
    aircrafts = state_data.get('aircrafts', {})

    # Find column names containing position coordinates
    x_cols = [col for col in survey_row.keys() if 'x' in col.lower()]
    y_cols = [col for col in survey_row.keys() if 'y' in col.lower()]
    print("Found x columns:", x_cols)
    print("Found y columns:", y_cols)

    # hUman x,yis col 17,18
    # agent x,y is col 19,20

    # Question 1_1: Human position (aircraft 1)
    human_pos = aircrafts.get('1', {}).get('position', (0, 0))
    human_x_col = next((col for col in x_cols if '1_1' in col), None)
    human_y_col = next((col for col in y_cols if '1_1' in col), None)

    print('\n### GRADING 1.1 ###')
    print('Human position:', human_pos)

    if human_x_col and human_y_col:
        grades.append(check_position(
            float(survey_row.get(human_x_col) or 0),
            float(survey_row.get(human_y_col) or 0),
            human_pos[0],
            human_pos[1]
        ))
    else:
        grades.append(0)

    # Question 2_1: Agent position (aircraft 0)
    agent_pos = aircrafts.get('0', {}).get('position', (0, 0))
    agent_x_col = next((col for col in x_cols if '2_1' in col), None)
    agent_y_col = next((col for col in y_cols if '2_1' in col), None)

    if agent_x_col and agent_y_col:
        grades.append(check_position(
            float(survey_row.get(agent_x_col) or 0),
            float(survey_row.get(agent_y_col) or 0),
            agent_pos[0],
            agent_pos[1]
        ))
    else:
        grades.append(0)

    # Q5: Agent health
    agent0_health = damage_to_health(aircrafts.get('0', {}).get('damage', 0))
    health_col = next((col for col in survey_row.keys() if 'Q5' in col), None)
    survey_health = float(survey_row.get(health_col) or 0)
    grades.append(1 if abs(agent0_health - survey_health) <= 3 else 0)

    # Q6: Human health
    agent1_health = damage_to_health(aircrafts.get('1', {}).get('damage', 0))
    health_col = next((col for col in survey_row.keys() if 'Q6' in col), None)
    survey_health = float(survey_row.get(health_col) or 0)
    grades.append(1 if abs(agent1_health - survey_health) <= 3 else 0)

    # Q13: Percentage of targets identified
    identified = game_state.get('identified_targets', 0)
    total_targets = 60
    actual_percentage = (identified / total_targets) * 100
    percent_col = next((col for col in survey_row.keys() if 'Q13' in col), None)
    survey_percentage = float(survey_row.get(percent_col) or 0)
    grades.append(1 if abs(actual_percentage - survey_percentage) <= 10 else 0)

    # Q14: Time remaining
    time_remaining = 245 - game_state.get('timestamp', 0) / 1000
    time_col = next((col for col in survey_row.keys() if 'Q14' in col), None)
    survey_time = float(survey_row.get(time_col) or 0)
    grades.append(1 if abs(time_remaining - survey_time) <= 10 else 0)

    # Q15: Agent mode
    mode_col = next((col for col in survey_row.keys() if 'Q15' in col), None)
    agent_mode = game_state.get('agent_mode', '')
    survey_mode = str(survey_row.get(mode_col) or '')
    grades.append(1 if agent_mode.lower() == survey_mode.lower() else 0)

    # Q11: Final targets identified
    final_col = next((col for col in survey_row.keys() if 'Q11' in col), None)
    survey_final = float(survey_row.get(final_col) or 0)
    grades.append(1 if abs(final_targets - survey_final) <= 2 else 0)

    print("Calculated grades:", grades)
    return grades


def process_surveys(data_folder: str, input_excel: str, output_excel: str):
    """Main function to process all surveys and generate grades."""
    # Load survey data
    survey_df = load_survey_data(input_excel)

    # Initialize results dictionary
    results = {}

    # Print diagnostic information
    print(f"Found {len(survey_df)} total survey responses")
    print(f"Unique subject IDs: {survey_df['subject_id'].unique()}")

    # Process each subject's surveys
    for subject_id in survey_df['subject_id'].unique():
        subject_grades = []
        subject_surveys = survey_df[survey_df['subject_id'] == subject_id]

        for _, survey in subject_surveys.iterrows():
            # Convert scenario and survey numbers, handling potential string values
            try:
                scenario = int(float(survey['scenario_number']))
                survey_num = int(float(survey['survey_number']))
            except (ValueError, TypeError):
                print(f"Warning: Invalid scenario/survey number for subject {subject_id}")
                continue

            # Calculate target time based on survey number
            target_time = survey_num * 60  # 60, 120, or 180 seconds

            # Find and load corresponding game state
            jsonl_file = find_jsonl_file(data_folder, str(subject_id), scenario)
            if jsonl_file:

                game_state = load_game_state(jsonl_file, target_time)
                print(game_state)
                final_targets = get_final_targets(jsonl_file)

                if game_state:
                    # Grade this survey
                    grades = grade_response(survey, game_state, final_targets)
                    subject_grades.extend(grades)

        results[subject_id] = subject_grades

    # Generate column names first
    question_types = ['Position_Human', 'Position_Agent', 'Health_Agent', 'Health_Human',
                      'Targets_Percent', 'Time_Remaining', 'Agent_Mode', 'Final_Targets']
    cols = []
    for round_num in range(1, 5):
        for survey_num in range(1, 4):
            for q_type in question_types:
                cols.append(f'Round{round_num}_Survey{survey_num}_{q_type}')

    # Create DataFrame with the right shape
    if not results:
        print("Warning: No results to process")
        result_df = pd.DataFrame(columns=cols)
    else:
        # Ensure all results have the same length as cols
        for subject_id in results:
            if len(results[subject_id]) < len(cols):
                results[subject_id].extend([0] * (len(cols) - len(results[subject_id])))
            elif len(results[subject_id]) > len(cols):
                results[subject_id] = results[subject_id][:len(cols)]

        result_df = pd.DataFrame.from_dict(results, orient='index', columns=cols)
    result_df.index.name = 'subject_id'

    # Save to Excel
    result_df.to_excel(output_excel)
    print(f"Grades saved to {output_excel}")

if __name__ == "__main__":
    data_folder = "sagat_grade_sample"  # Folder containing all JSONL files
    input_excel_file = "sagat_qualtrics_export_test2.xlsx"
    output_excel_file = "sagat_grades_test.xlsx"

    process_surveys(data_folder, input_excel_file, output_excel_file)