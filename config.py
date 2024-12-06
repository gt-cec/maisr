"""To set up experiment run:"""

# 1. Enter a subject_id:
subject_id = '999'
user_group = 'test' # 'test', 'control' 'card' 'in situ'
# run_order = [1,2,3,4] # TODO Implement this

# 2. Make sure log_data = True in order to save run data to a json file in ./experiment_data
log_data = False


# 3. Setup render window parameters if needed
x = 1680+235    # Pixel location of right edge of the pygame window. Default = 235 (renders in center of a 1920x1080 screen if gameboard is configured to 1450px)
y = -3          # Pixel location of top edge of the pygame window


# 4. (Optional) If necessary, modify which configs are loaded based on the user_group. You will probably skip this.

config_dict = { # TEMP FOR SUBJ 102
               'test': ['./config_files/'+'model_card_configs/modelcard_scenario1_config.json', './config_files/'+'model_card_configs/modelcard_scenario2_config.json', './config_files/'+'model_card_configs/modelcard_scenario3_config.json', './config_files/'+'model_card_configs/modelcard_scenario4_config.json'],
               'control': ['./config_files/'+'training_control_config.json', './config_files/'+'model_card_configs/modelcard_scenario3_config.json', './config_files/'+'model_card_configs/modelcard_scenario4_config.json', './config_files/'+'model_card_configs/modelcard_scenario2_config.json', './config_files/'+'model_card_configs/modelcard_scenario1_config.json'],
               'card': ['./config_files/'+'training_control_config.json', './config_files/'+'model_card_configs/modelcard_scenario1_config.json', './config_files/'+'model_card_configs/modelcard_scenario3_config.json', './config_files/'+'model_card_configs/modelcard_scenario4_config.json', './config_files/'+'model_card_configs/modelcard_scenario2_config.json'],
               'in situ':['./config_files/'+'training_insitu_config.json', './config_files/'+'model_card_configs/modelcard_scenario1_config.json', './config_files/'+'model_card_configs/modelcard_scenario2_config.json', './config_files/'+'model_card_configs/modelcard_scenario3_config.json', './config_files/'+'model_card_configs/modelcard_scenario4_config.json']
               }

# config_dict = {
#                'test': ['./config_files/'+'model_card_configs/modelcard_scenario1_config.json', './config_files/'+'model_card_configs/modelcard_scenario2_config.json', './config_files/'+'model_card_configs/modelcard_scenario3_config.json', './config_files/'+'model_card_configs/modelcard_scenario4_config.json'],
#                'control': ['./config_files/'+'training_control_config.json', './config_files/'+'model_card_configs/modelcard_scenario1_config.json', './config_files/'+'model_card_configs/modelcard_scenario2_config.json', './config_files/'+'model_card_configs/modelcard_scenario3_config.json', './config_files/'+'model_card_configs/modelcard_scenario4_config.json'],
#                'card': ['./config_files/'+'training_control_config.json', './config_files/'+'model_card_configs/modelcard_scenario1_config.json', './config_files/'+'model_card_configs/modelcard_scenario2_config.json', './config_files/'+'model_card_configs/modelcard_scenario3_config.json', './config_files/'+'model_card_configs/modelcard_scenario4_config.json'],
#                'in situ':['./config_files/'+'training_insitu_config.json', './config_files/'+'model_card_configs/modelcard_scenario1_config.json', './config_files/'+'model_card_configs/modelcard_scenario2_config.json', './config_files/'+'model_card_configs/modelcard_scenario3_config.json', './config_files/'+'model_card_configs/modelcard_scenario4_config.json']
#                }