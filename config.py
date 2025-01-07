"""To set up experiment run:"""

# 1. Enter a subject_id:
subject_id = '907'
user_group = 'test' # 'test', 'control' 'card' 'in situ'
run_order = 1 # 1,2,3, or 4


# 2. Make sure log_data = True (run data will be saved to a json file in ./experiment_data)
log_data = True


# 3. Setup render window parameters if needed
x = 0#1680+235    # Pixel location of right edge of the pygame window. Default = 235 (renders in center of a 1920x1080 screen if gameboard is configured to 1450px)
y = -3          # Pixel location of top edge of the pygame window



# 4. (Optional) If necessary, modify which configs are loaded based on the user_group. You will probably skip this.
training_configs = {
    'test': './config_files/model_card_configs/modelcard_scenario1_config.json',
    'control': './config_files/training_control_config.json',
    'card': './config_files/training_control_config.json',
    'in situ': './config_files/training_insitu_config.json'}

scenario_configs = [
    './config_files/model_card_configs/modelcard_scenario1_config.json',
    './config_files/model_card_configs/modelcard_scenario2_config.json',
    './config_files/model_card_configs/modelcard_scenario3_config.json',
    './config_files/model_card_configs/modelcard_scenario4_config.json']

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
    for group in ['test', 'control', 'card', 'in situ']}