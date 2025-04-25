################################################### FOR MANUAL SETUP ###################################################
"""Modify these variables if you want to run main.py directly instead of using run.bat"""

# Enter a subject_id:
subject_id = '999' # Used to name log files, and passed into qualtrics survey metadata
user_group = 'test' # 'test', 'control' 'card' 'in-situ'
run_order = 1 # Int, 1,2,3, or 4
round_number = '0' # String 0,1,2,3,4 (Used to start from a round other than 0, e.g. if you need to restart during a study.

# SAGAT surveys
surveys_enabled = True # TODO Should add this to the run.bat args too
times = [65.0, 125.0, 185.0]

# Set whether to log data (run data will be saved to a json file in ./experiment_data)
log_data = 'y' # 'y' or 'n' (string)

# Modify pygame render window parameters if needed
x = 235 #1680+235    # Pixel location of right edge of the pygame window. Default = 235 (renders in center of a 1920x1080 screen if gameboard is configured to 1450px)
y = -3          # Pixel location of top edge of the pygame window


#################################################### CONFIG LOADING ####################################################
"""These lines specific which config files are loaded for each user_group, and which order they are played in based on 
run_order. You do not need to modify this unless you want to load different config files from /config_files/ or change
how they are ordered. """


training_configs = {
    'test': './config_files/modelcard_scenario1_config.json',
    'control': './config_files/training_control_config.json',
    'card': './config_files/training_control_config.json',
    'in-situ': './config_files/training_insitu_config.json'}

scenario_configs = [
    './config_files/modelcard_scenario1_config.json',
    './config_files/modelcard_scenario2_config.json',
    './config_files/modelcard_scenario3_config.json',
    './config_files//modelcard_scenario4_config.json']

def get_ordered_configs(user_group, run_order):
    # Get the appropriate training config
    training_config = training_configs[user_group]

    # Define scenario ordering based on run_order
    order_mappings = {
        1: [0, 1, 2, 3],  # Original order: 1,2,3,4
        2: [1, 2, 3, 0],  # Start with 2: 2,3,4,1
        3: [2, 0, 1, 3],  # Start with 3: 3,1,2,4
        4: [3, 2, 0, 1]  # Start with 4: 4,3,1,2
    }

    # Reorder scenario configs based on run_order
    if user_group == 'test':
        # Test group only gets scenario configs, no training
        ordered_scenarios = [scenario_configs[i] for i in order_mappings[run_order]]
        return ordered_scenarios
    else:
        # Other groups get training + scenarios
        ordered_scenarios = [scenario_configs[i] for i in order_mappings[run_order]]
        return [training_config] + ordered_scenarios


# Generate the config dictionary based on run_order
config_dict = {
    group: get_ordered_configs(group, run_order)
    for group in ['test', 'control', 'card', 'in-situ']}

