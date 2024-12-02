"""To set up experiment run:"""
# 1. Enter a subject_id:
subject_id = '999'
user_group = 'test' # 'test', 'control' 'card' 'video'


# 2. Modify config_filename:
config_1 = 'model_card_configs/modelcard_scenario1_config.json'#'training_config.json'
config_2 = 'model_card_configs/modelcard_scenario1_config.json'
config_3 = 'model_card_configs/modelcard_scenario2_config.json'
config_4 = 'model_card_configs/modelcard_scenario3_config.json'
config_5 = 'model_card_configs/modelcard_scenario4_config.json'


# 3. Set whether you want to save run data to a json file in ./experiment_data:
log_data = False

# 4. Setup render window parameters if needed
x = 235 # Pixel location of right edge of the pygame window. Default = 235 (renders in center of a 1920x1080 screen if gameboard is configured to 1450px)
y = 0 # Pixel location of top edge of the pygame window