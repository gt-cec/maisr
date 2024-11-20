"""To set up experiment run:"""
# 1. Enter a subject_id:
subject_id = '000'


# 2. Modify config_filename:

#config_filename = 'training_config.json'
config_filename = 'model_card_configs/modelcard_scenario1_config.json'
#config_filename = 'model_card_configs/modelcard_scenario2_config.json'
#config_filename = 'model_card_configs/modelcard_scenario3_config.json'
#config_filename = 'model_card_configs/modelcard_scenario4_config.json'


# 3. Set whether you want to save run data to a json file in ./experiment_data:
log_data = False



"""
Config filenames (all are in ./config_files/):

'default_config.json'
'training_config.json' : For training runs. Agent is not rendered, and only 10 ships are spawned. 

"""