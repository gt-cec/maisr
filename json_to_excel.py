import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def convert_json_to_excel(json_file_path, excel_file_path):
    """
    Convert JSON timestep data to Excel format where each row is a timestep.

    Args:
        json_file_path (str): Path to the input JSON file
        excel_file_path (str): Path to the output Excel file
    """

    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract episode number
    episode = data['episode']
    timesteps = data['timesteps']

    # Prepare data for DataFrame
    rows = []

    for timestep_data in timesteps:
        row = {
            'episode': episode,
            'timestep': timestep_data['timestep'],
            'reward': timestep_data['reward']
        }

        # Add observation columns (obs_0, obs_1, obs_2, etc.)
        observation = timestep_data['observation']
        for i, obs_value in enumerate(observation):
            row[f'obs_{i}'] = obs_value

        # Add raw actions
        raw_actions = timestep_data['raw actions']
        for i, action in enumerate(raw_actions):
            row[f'raw_action_{i}'] = action

        # Add processed actions
        processed_actions = timestep_data['processed actions']
        for i, action in enumerate(processed_actions):
            row[f'processed_action_{i}'] = action

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns for better readability
    basic_cols = ['episode', 'timestep', 'reward']
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    raw_action_cols = [col for col in df.columns if col.startswith('raw_action_')]
    processed_action_cols = [col for col in df.columns if col.startswith('processed_action_')]

    # Reorder columns: basic info, then observations, then actions
    column_order = basic_cols + obs_cols + raw_action_cols + processed_action_cols
    df = df[column_order]

    # Save to Excel
    df.to_excel(excel_file_path, index=False, sheet_name='Timesteps')

    print(f"Successfully converted JSON to Excel!")
    print(f"Output file: {excel_file_path}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(
        f"Columns: {len(basic_cols)} basic + {len(obs_cols)} observation + {len(raw_action_cols)} raw action + {len(processed_action_cols)} processed action")


def convert_json_to_excel_with_summary(json_file_path, excel_file_path):
    """
    Enhanced version that creates multiple sheets:
    - Main data sheet with all timesteps
    - Summary sheet with basic statistics
    """

    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    episode = data['episode']
    timesteps = data['timesteps']

    # Prepare main data
    rows = []
    for timestep_data in timesteps:
        row = {
            'episode': episode,
            'timestep': timestep_data['timestep'],
            'reward': timestep_data['reward']
        }

        # Add observations
        observation = timestep_data['observation']
        for i, obs_value in enumerate(observation):
            row[f'obs_{i}'] = obs_value

        # Add actions
        raw_actions = timestep_data['raw actions']
        for i, action in enumerate(raw_actions):
            row[f'raw_action_{i}'] = action

        processed_actions = timestep_data['processed actions']
        for i, action in enumerate(processed_actions):
            row[f'processed_action_{i}'] = action

        rows.append(row)

    # Create main DataFrame
    df_main = pd.DataFrame(rows)

    # Reorder columns
    basic_cols = ['episode', 'timestep', 'reward']
    obs_cols = [col for col in df_main.columns if col.startswith('obs_')]
    raw_action_cols = [col for col in df_main.columns if col.startswith('raw_action_')]
    processed_action_cols = [col for col in df_main.columns if col.startswith('processed_action_')]

    column_order = basic_cols + obs_cols + raw_action_cols + processed_action_cols
    df_main = df_main[column_order]

    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Episode',
            'Total Timesteps',
            'Total Reward',
            'Average Reward',
            'Max Reward',
            'Min Reward',
            'Observation Vector Size',
            'Raw Actions Size',
            'Processed Actions Size'
        ],
        'Value': [
            episode,
            len(timesteps),
            df_main['reward'].sum(),
            df_main['reward'].mean(),
            df_main['reward'].max(),
            df_main['reward'].min(),
            len(obs_cols),
            len(raw_action_cols),
            len(processed_action_cols)
        ]
    }
    df_summary = pd.DataFrame(summary_data)

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        df_main.to_excel(writer, sheet_name='Timestep_Data', index=False)

    print(f"Successfully converted JSON to Excel with summary!")
    print(f"Output file: {excel_file_path}")
    print(f"Main data shape: {df_main.shape[0]} rows × {df_main.shape[1]} columns")
    print(f"Created 2 sheets: 'Summary' and 'Timestep_Data'")


# Example usage
if __name__ == "__main__":
    # Basic conversion
    # convert_json_to_excel('short.json', 'timestep_data.xlsx')

    # Enhanced conversion with summary
    convert_json_to_excel_with_summary('oar_6envs_obs-relative_act-continuous-normalized_lr-5e-05_bs-128_g-0.998_fs-1_ppoupdates-2048_curriculum-Truerew-wtn-0.005_rew-prox-0.0015_rew-timepenalty--0.0_0530_1041_ep100.json', 'timestep_data_with_summary.xlsx')