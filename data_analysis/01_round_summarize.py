import matplotlib.pyplot as plt
import json
import numpy as np
from scipy.interpolate import interp1d
import os



def extract_gameplan_history(filename):
    """Extract gameplan command history from the log file"""
    with open(filename, 'r') as f:
        content = f.read()

    # Find the gameplan history in the final stats
    try:
        # Look for the gameplan history list at the end of the file
        if '"gameplan_command_history": ' in content:
            history_start = content.find('"gameplan_command_history": ') + len('"gameplan_command_history": ')
            history_end = content.find('], "total_gameplan_commands"')
            if history_end == -1:  # If not found, try finding just the end bracket
                history_end = content.find(']', history_start) + 1
            history_str = content[history_start:history_end + 1]
            return json.loads(history_str)
    except:
        print("Error parsing gameplan history")
        return []


def extract_metadata(filename):
    """Extract subject ID, user group, and other metadata from the log file"""
    metadata = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:4]):  # Check first 4 lines for metadata
            if 'subject_id:' in line:
                metadata['subject_id'] = line.split(':')[1].strip()[0:-1]
            elif 'user_group:' in line:
                metadata['user_group'] = line.split(':')[1].strip()[0:-1]
            elif '"game configuration:' in line:
                # Extract scenario number from config path
                config_path = line.split(':')[1].strip().strip('"')
                if 'scenario' in config_path:
                    try:
                        scenario_num = config_path.split('scenario')[1][0]
                        metadata['scenario'] = scenario_num
                    except:
                        metadata['scenario'] = 'unknown'
    return metadata

def analyze_game_log(filename, save_plot = False):
    metadata = extract_metadata(filename)
    gameplan_history = extract_gameplan_history(filename)

    # Process score history from state updates
    times = []
    scores = []
    human_health = None
    agent_health = None
    targets_identified = None
    threat_types_identified = None
    mouse_clicks = 0
    final_time = None

    # Extract score data and other metrics from the game state history
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('"game configuration'):
                continue
            try:
                data = json.loads(line)
                if 'type' in data:
                    if data['type'] == 'state':
                        times.append(data['timestamp'])
                        scores.append(data['game_state']['score'])
                        final_time = data['timestamp']
                        targets_identified = data['game_state']['identified_targets']
                        weapons_identified = data['game_state']['identified_threat_types']
                        agent_health = 4 - data['game_state']['agent0_damage']/25 #100 - data['game_state']['agent0_damage']
                        human_health = 4 - data['game_state']['agent1_damage']/25 # 100 - data['game_state']['agent1_damage']
                    elif data['type'] == 'mouse_event':
                        mouse_clicks += 1
            except json.JSONDecodeError:
                continue

    # Create combined plot
    plt.figure(figsize=(15, 8))

    # Plot score history
    plt.plot(times, scores, 'b-', label='Score', linewidth=2)

    # Create legend text with metadata
    # legend_text = [
    #     f'Score',
    #     f'Subject ID: {metadata.get("subject_id", "unknown")}',
    #     f'User Group: {metadata.get("user_group", "unknown")}',
    #     f'Round: {metadata.get("scenario", "unknown")}'
    # ]

    if scores and gameplan_history:
        # Create interpolation function for scores
        score_interp = interp1d(times, scores, kind='linear')

        # Plot command markers and labels
        for time, command in gameplan_history:
            if time <= max(times):
                # Get interpolated score at command time
                score_at_time = float(score_interp(time))

                # Plot command marker
                plt.plot(time, score_at_time, 'r.', markersize=10)

                # Add command label above the line
                # Calculate vertical offset based on score range
                score_range = max(scores) - min(scores)
                offset = score_range * 0.03  # 3% of score range

                plt.text(time, score_at_time + offset, command,
                         ha='center', va='bottom', rotation=45,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.xlabel('Time (seconds)')
    plt.ylabel('Score')
    plt.title('Team Score Over Time with Game Commands')
    plt.title(f'Subject {metadata.get('subject_id', '000')} ({metadata.get("user_group", "unknown")} group) - Round {metadata.get('scenario', '0')}')
    plt.grid(True)

    # Add legend with metadata
    #plt.legend(legend_text, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.margins(y=0.15)
    plt.tight_layout()  # Adjust layout to make room for legend

    plt.show()

    # Save plot with formatted filename
    if save_plot:

        plot_filename = f"MAISR_subj{metadata.get('subject_id', '000')}_round_{metadata.get('scenario', '0')}.png"
        plot_filename = os.path.join(os.getcwd(), plot_filename)  # Create proper path
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        print(f"Plot saved as: {plot_filename}")
        plt.close()  # Close the figure to free memory

    # Create summary dictionary
    summary = {
        "final_score": scores[-1] if scores else None,
        "human_health": human_health,
        "agent_health": agent_health,
        "targets_identified": targets_identified,
        "weapons_identified": threat_types_identified,
        "total_gameplan_commands": len(gameplan_history),
        "total_mouse_clicks": mouse_clicks,
        "time_remaining": 240 - final_time if final_time is not None else None
    }

    print("\nGame Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


# Example usage
if __name__ == "__main__":
    summary = analyze_game_log("sample_data_3.jsonl",save_plot = False)
