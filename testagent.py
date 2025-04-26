from agents import *
import sys
import os
import ctypes
import torch
from env import MAISREnv
from gui import *
from utility.data_logging import GameLogger, load_env_config
from config import x, y, config_dict, run_order, surveys_enabled, times
#from autonomous_policy import AutonomousPolicy
from a2c_policy import MAISRActorCritic
import webbrowser

class SAGAT:
    def __init__(self, env, times, subject_id, user_group, round_number):
        self.env = env
        self.times = [time + random.uniform(-10, 10) for time in times] # list of times to launch survey. Should be 3. To induce randomness, this operation randomly samples a time from 20 seconds around the specified time
        self.surveys_enabled = surveys_enabled
        self.subject_id = subject_id
        self.user_group = user_group
        self.round_number = round_number
        self.survey1_launched, self.survey2_launched, self.survey3_launched = False, False, False

    def check(self):
        if times[0] - 1.0 < env.display_time/1000 < times[0] + 1.0 and not self.survey1_launched:
            print(f'Time = {env.display_time/1000}, survey 1 triggered')
            webbrowser.open_new_tab(
                'https://gatech.co1.qualtrics.com/jfe/form/SV_egiLZSvblF8SVO6?subject_id=' + str(
                    self.subject_id) + '&scenario_number=' + str(self.round_number) + '&user_group=' + str(
                    self.user_group) + '&survey_number=1')
            self.survey1_launched = True
            env.pause(pygame.MOUSEBUTTONDOWN)

        elif times[1] - 1.0 < env.display_time/1000 < times[1] + 1.0 and not self.survey2_launched:
            print(f'Time = {env.display_time / 1000}, survey 2 triggered')
            webbrowser.open_new_tab(
                'https://gatech.co1.qualtrics.com/jfe/form/SV_egiLZSvblF8SVO6?subject_id=' + str(
                    self.subject_id) + '&scenario_number=' + str(self.round_number) + '&user_group=' + str(
                    self.user_group) + '&survey_number=2')
            self.survey2_launched = True
            env.pause(pygame.MOUSEBUTTONDOWN)

        elif times[2] - 1.0 < env.display_time/1000 < times[2] + 1.0 and not self.survey3_launched:
            print(f'Time = {env.display_time / 1000}, survey 3 triggered')
            webbrowser.open_new_tab(
                'https://gatech.co1.qualtrics.com/jfe/form/SV_egiLZSvblF8SVO6?subject_id=' + str(
                    self.subject_id) + '&scenario_number=' + str(self.round_number) + '&user_group=' + str(
                    self.user_group) + '&survey_number=3')
            self.survey3_launched = True
            env.pause(pygame.MOUSEBUTTONDOWN)

if __name__ == "__main__":

    checkpoint_path = 'checkpoints/best_model_4_26.pt'
    config_path = './config_files/rl_training_config.json'

    if len(sys.argv) == 1:
        print('No args specified, loading parameters from config.py')
        from config import subject_id, user_group, log_data, x, y, round_number

    elif len(sys.argv) < 5:
        print("Missing args, run as: python main.py subject_id user_group starting_round_number log_data")
        sys.exit()

    else:
        subject_id = sys.argv[1]
        user_group = sys.argv[2]
        round_number = sys.argv[3]
        log_data = sys.argv[4]

    if not subject_id.isdigit():
        print("Invalid subject ID: >" + subject_id + "<")
        sys.exit()

    if user_group not in ["test", "card", "control", "in-situ"]:
        print("Invalid user group: >" + user_group + "<")
        sys.exit()

    if round_number not in ["0", "1", "2", "3", "4"] or not round_number.isdigit():
        print("Invalid round number: " + ">" + round_number + "<")
        sys.exit()
    round_number = int(round_number)

    if log_data == 'y':
        log_data = True
    elif log_data == 'n':
        log_data = False
    else:
        print("Invalid input for log data")
        sys.exit()

    print(f"\nStarting MAISR environment (subject_id = {subject_id}, group = {user_group}, data logging = {log_data})")

    render = "headless" not in sys.argv
    reward_type = 'balanced-sparse' # TODO move into a config file



    config_list = config_dict[user_group]
    total_games = 5 # Number of games to run
    game_count = 0 # Used to track how many games have been completed so far

    while round_number < total_games:
        #config = config_list[round_number]
        env_config = load_env_config(config_path)

        if render:
            print("Starting in PyGame mode")
            pygame.init()
            clock = pygame.time.Clock()
            ctypes.windll.user32.SetProcessDPIAware()  # Disables display scaling so the game fits on small, high-res monitors
            window_width, window_height = env_config['window size'][0], env_config['window size'][1]
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"
            window = pygame.display.set_mode((window_width, window_height),flags=pygame.NOFRAME)
            env = MAISREnv(env_config, window, clock=clock, render_mode='human',
                           reward_type=reward_type, obs_type='vector', action_type='continuous',
                           subject_id=subject_id,user_group=user_group,round_number=round_number)

        else:
            print("Starting in headless mode")
            pygame.init()
            clock = pygame.time.Clock()
            pygame.font.init()
            env = MAISREnv(env_config, None, render_mode='none',
                           reward_type=reward_type, obs_type='vector', action_type='continuous',
                           subject_id=subject_id, user_group=user_group, round_number=round_number)


        # Create the model with the same architecture as during training
        obs_dim = env.observation_space.shape[0]
        act_dim_continuous = 2  # x, y coordinates for waypoint
        act_dim_discrete = 1  # ID method options (changed from 3 to 1 to match training model)
        gameboard_size = env.config["gameboard size"]

        print(
            f"Creating model with obs_dim={obs_dim}, act_dim_continuous={act_dim_continuous}, act_dim_discrete={act_dim_discrete}")
        agent0_policy = MAISRActorCritic(obs_dim, act_dim_continuous, act_dim_discrete, gameboard_size)

        # Load the model checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        agent0_policy.load_state_dict(checkpoint['model_state_dict'])
        agent0_policy.eval()  # Set to evaluation mode

        print(f"Loaded model from checkpoint: {checkpoint_path}")


        agent0_id = env.num_ships  # Hack to dynamically get agent IDs

        if log_data:
            game_logger = GameLogger(subject_id, 'training_test', user_group, round_number, run_order)
            game_logger.initial_log()
            button_handler = ButtonHandler(env, agent0_policy, game_logger, log_data=True)

        else: button_handler = ButtonHandler(env, agent0_policy, log_data=False)

        game_count += 1
        state = env.reset()  # reset the environment
        terminated, truncated = False, False  # flag for when the run is complete
        agent1_waypoint = (0, 0)

        agent_log_info = {'waypoint': 'None', 'priority mode': 'None', 'search type': 'None', 'search area': 'None'}
        agent_action = {'waypoint':(0,0), 'id_method':0}

        hold_commanded = False

        while not (terminated or truncated):  # main game loop
            actions = []  # use agent policies to get actions as a list of tuple [(agent index, waypoint)]

            # Handle human actions (mouse clicks)
            ev = pygame.event.get() # TODO Remove command buttons
            for event in ev:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_F1:
                        if log_data: game_logger.final_log(button_handler.gameplan_command_history, env)
                        pygame.quit()
                    if event.key == pygame.K_SPACE:
                        env.pause(pygame.MOUSEBUTTONDOWN)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_position = pygame.mouse.get_pos()
                    time_sec = float(env.display_time) / 1000

                    _, human_waypoint = button_handler.handle_mouse_click(mouse_position, time_sec)

                    human_action = {'waypoint': human_waypoint, 'id_method': 0}
                    new_human_action = True # TODO placeholder hack since human_action is never None because of above

                    if human_action and new_human_action:
                        #print(f'MAIN: HUMAN {(env.aircraft_ids[1],human_action)}')
                        actions.append((env.human_idx,human_action))
                        agent1_waypoint = human_action
                        new_human_action = False

            # actions: List of (agent_id, action) tuples, where action = dict('waypoint': (x,y), 'id_method': 0, 1, or 2')
            if not hold_commanded:
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Get action from the model
                with torch.no_grad():
                    agent_action = agent0_policy.act(state_tensor)
                    print(f'Agent chose action {agent_action}')

                if isinstance(agent_action, dict):
                    agent_action = {
                        'waypoint': agent_action['waypoint'],
                        'id_method': agent_action['id_method']
                    }
                else:
                    # If the model returns waypoints directly
                    agent_action = {
                        'waypoint': agent_action,
                        'id_method': 0  # Default ID method
                    }
                actions.append((env.aircraft_ids[0], agent_action))
                hold_commanded = False


            if env.init or pygame.time.get_ticks() > env.start_countdown_time:
                observation, reward, terminated, truncated, info = env.step(actions)  # step through the environment

            if env.init: env.init = False
            if env.render_mode == 'human':
                env.render()

        if terminated or truncated:
            done_time = pygame.time.get_ticks()
            if log_data:
                game_logger.log_state(env, env.display_time,agent1_waypoint,agent_log_info)
                game_logger.final_log(button_handler.get_command_history(), env)

            if env.render_mode == 'human':
                waiting_for_key = True
                while waiting_for_key:
                    env.render()  # Keep rendering while waiting
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                            if (pygame.time.get_ticks()-done_time)/1000 > 1.5: # Cooldown to prevent accidentally clicking continue
                                waiting_for_key = False
                                break
                        elif event.type == pygame.QUIT:
                            if log_data:
                                game_logger.final_log(button_handler.get_command_history(), env)
                            pygame.quit()
                            sys.exit()

        round_number += 1
        if env.render_mode == 'human':
            print("Game complete:", game_count)
            pygame.quit()

    print("ALL GAMES COMPLETE")

    if surveys_enabled:
        webbrowser.open_new_tab('https://gatech.co1.qualtrics.com/jfe/form/SV_3ITZnNRRBqioKR8?subject_id='+str(subject_id)+'&user_group='+str(user_group))