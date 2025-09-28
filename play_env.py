import ctypes

import numpy as np
import pygame
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

from base_env import MaisrEnv
from utility.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.config_management import load_env_config
from utility.league_management import (
    PolicySelector, ConfigurableHeuristicTeammate,
    LocalSearch, GoToNearestThreat, ChangeRegions, TargetSearchLocalTSP
)


def load_vecnormalize_wrapper(vecnorm_path, env):
    """Load saved VecNormalize wrapper with stats from training and apply it to the new environment."""
    print(f"Loading VecNormalize stats from: {vecnorm_path}")
    vec_normalize = VecNormalize.load(vecnorm_path, venv=env)
    vec_normalize.training = False  # Disable further normalization updates
    vec_normalize.norm_reward = False
    return vec_normalize


def make_wrapped_env(config, clock, window, agent_appearance, tag='play_env'):
    """Create wrapped environment"""

    def _init():
        base_env = MaisrEnv(
            config=config,
            clock=clock,
            window=window,
            render_mode='human',
            run_name=f'play_env_{tag}',
            tag=tag,
            agent_appearance=agent_appearance,
            running_experiment=False
        )

        wrapped_env = MaisrLocalSearchWrapper(
            base_env,
            config['obs_noise_std_localsearch'],
            local_search_policy=None,
            go_to_highvalue_policy=None,
            change_region_subpolicy=None,
            evade_policy=None,
            teammate_policy=None,
        )

        wrapped_env.reset()
        return wrapped_env

    return _init

class HumanController:
    """Handles human input for subpolicy selection via keyboard and mouse clicks"""

    def __init__(self, env):
        self.env = env
        self.current_subpolicy = 0  # Default to local search
        self.last_click_time = 0
        self.click_cooldown = 200  # milliseconds between clicks
        self.custom_waypoint = None
        self.use_custom_waypoint = False

    def handle_click(self, mouse_pos):
        """Convert mouse click to custom waypoint and switch to custom mode"""
        current_time = pygame.time.get_ticks()

        # Prevent rapid clicking
        if current_time - self.last_click_time < self.click_cooldown:
            return None

        self.last_click_time = current_time

        # Get click position relative to game board
        click_x, click_y = mouse_pos
        gameboard_size = self.env.envs[0].env.config['gameboard_size']

        # Check if click is within gameboard bounds
        if 0 <= click_x <= gameboard_size and 0 <= click_y <= gameboard_size:
            # Convert from screen coordinates to game coordinates
            map_half_size = gameboard_size / 2
            game_x = click_x - map_half_size
            game_y = click_y - map_half_size

            # Store the waypoint for custom control
            #self.custom_waypoint = np.array([game_x / map_half_size, game_y / map_half_size], dtype=np.float32)
            self.custom_waypoint = np.array([game_x, game_y], dtype=np.float32)
            #print(f'Custom waypoint = {self.custom_waypoint}')
            self.env.human_custom_waypoint = (float(self.custom_waypoint[0]),float(self.custom_waypoint[1]))
            self.use_custom_waypoint = True

            #print(f"Human clicked at screen ({click_x}, {click_y}) -> game waypoint ({game_x:.1f}, {game_y:.1f})")
            #print("Switched to custom waypoint mode")
            return True

        return False

    def handle_keypress(self, key):
        """Handle keyboard input for subpolicy selection"""
        key_to_subpolicy = {
            pygame.K_1: 0,  # Local search
            pygame.K_2: 1,  # Goto NW
            pygame.K_3: 2,  # Go to threat
            pygame.K_4: 3,  # Hold
            pygame.K_5: 4,  # Goto NE
            pygame.K_6: 5,  # Goto SE
            pygame.K_7: 6,  # Goto SW
            pygame.K_8: 7,  # Custom waypoint mode
        }

        if key in key_to_subpolicy:
            self.current_subpolicy = key_to_subpolicy[key]
            if self.current_subpolicy == 7:
                self.use_custom_waypoint = True
            else:
                self.use_custom_waypoint = False

            subpolicy_names = ["Local Search", "Goto NW", "Go to Threat", "Hold",
                               "Goto NE", "Goto SE", "Goto SW", "Custom Waypoint"]
            print(f"Human selected subpolicy: {subpolicy_names[self.current_subpolicy]}")
            return True

        return False

    def get_current_action(self):
        """Get the current action for the human player"""
        return self.current_subpolicy

    def should_override_waypoint(self):
        """Check if we should override the agent's waypoint with custom waypoint"""
        return self.use_custom_waypoint and self.custom_waypoint is not None

    def get_custom_waypoint(self):
        """Get the custom waypoint coordinates"""
        return self.custom_waypoint

# class HumanDirectController:
#     """Handles human input via mouse clicks only - direct waypoint control"""
#
#     def __init__(self, env):
#         self.env = env
#         self.last_click_time = 0
#         self.click_cooldown = 100  # milliseconds between clicks
#         self.current_waypoint = None
#         self.default_waypoint = None
#
#     def handle_click(self, mouse_pos):
#         """Convert mouse click to waypoint"""
#         current_time = pygame.time.get_ticks()
#
#         # Prevent rapid clicking
#         if current_time - self.last_click_time < self.click_cooldown:
#             return False
#
#         self.last_click_time = current_time
#
#         # Get click position relative to game board
#         click_x, click_y = mouse_pos
#         gameboard_size = self.env.envs[0].env.config['gameboard_size']
#
#         # Check if click is within gameboard bounds
#         if 0 <= click_x <= gameboard_size and 0 <= click_y <= gameboard_size:
#             # Convert from screen coordinates to game coordinates
#             map_half_size = gameboard_size / 2
#             game_x = click_x - map_half_size
#             game_y = click_y - map_half_size
#
#             # Store the waypoint
#             self.current_waypoint = (float(game_x), float(game_y))
#             print(f"Human clicked at screen ({click_x}, {click_y}) -> game waypoint ({game_x:.1f}, {game_y:.1f})")
#             return True
#
#         return False
#
#     def handle_keypress(self, key):
#         """No keyboard handling needed for direct control"""
#         return False
#
#     def get_current_waypoint(self):
#         """Get the current waypoint coordinates"""
#         if self.current_waypoint is not None:
#             return self.current_waypoint
#         else:
#             # If no waypoint set, return current position (stay put)
#             if self.default_waypoint is None:
#                 # Set default to current agent position
#                 try:
#                     agent = self.env.envs[0].env.agents[self.env.envs[0].env.aircraft_ids[0]]
#                     self.default_waypoint = (float(agent.x), float(agent.y))
#                 except:
#                     self.default_waypoint = (0.0, 0.0)
#             return self.default_waypoint
#
#     def has_waypoint(self):
#         """Check if a waypoint has been set"""
#         return self.current_waypoint is not None


def create_heuristic_agent(agent_type, config):
    """Create a heuristic-based agent"""

    # Create subpolicies
    local_search_policy = LocalSearch()
    go_to_threat_policy = GoToNearestThreat()
    change_region_policy = ChangeRegions()

    # Define different heuristic agent configurations
    if agent_type == "conservative":
        heuristic_agent = PolicySelector(
            mode_selector="heuristic",
            risk_tolerance="low",
            spatial_coord=False
        )
        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=False,
            search_method="greedy"
        )
    elif agent_type == "aggressive":
        heuristic_agent = PolicySelector(
            mode_selector="heuristic",
            risk_tolerance="high",
            spatial_coord=False
        )
        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=False,
            search_method="greedy"
        )
    elif agent_type == "coordinated":
        heuristic_agent = PolicySelector(
            mode_selector="heuristic",
            risk_tolerance="medium",
            spatial_coord=True
        )
        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=True,
            search_method="clusters"
        )
    elif agent_type == "greedy":
        heuristic_agent = PolicySelector(
            mode_selector="heuristic",
            risk_tolerance="max_greedy",
            spatial_coord=False
        )
        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=False,
            search_method="greedy"
        )
    else:
        # Default conservative
        heuristic_agent = PolicySelector(
            mode_selector="heuristic",
            risk_tolerance="low",
            spatial_coord=False
        )
        target_search_policy = TargetSearchLocalTSP(
            search_radius=1000,
            spatial_coord=False,
            search_method="greedy"
        )

    # Create the teammate policy
    teammate = ConfigurableHeuristicTeammate(
        env=None,
        local_search_policy=target_search_policy,
        go_to_highvalue_policy=go_to_threat_policy,
        change_region_subpolicy=change_region_policy,
        mode_selector_agent=heuristic_agent,
        use_collision_avoidance=False,
        action_stability="stable",
        decision_speed="fast"
    )

    teammate.name = f"Heuristic_{agent_type}"
    return teammate


def run_episode(env, agent0_controller, agent1_controller, config, window, clock, tick_rate):
    """Run a single episode with the configured agents"""

    print(f"\n{'=' * 50}")
    print(f"Starting episode")
    print(f"Agent 0: {type(agent0_controller).__name__}")
    print(f"Agent 1: {type(agent1_controller).__name__}")
    print(f"{'=' * 50}")

    obs = env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    paused = False

    base_env = env.envs[0].env

    while not done:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True, episode_reward, step_count
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True, episode_reward, step_count
                elif event.key == pygame.K_p:
                    paused = not paused
                    print("Game paused" if paused else "Game resumed")
                else:
                    # Pass keypress to human controllers if applicable
                    if hasattr(agent0_controller, 'handle_keypress'):
                        agent0_controller.handle_keypress(event.key)
                    if hasattr(agent1_controller, 'handle_keypress'):
                        agent1_controller.handle_keypress(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Pass click to human controllers if applicable
                    if hasattr(agent0_controller, 'handle_click'):
                        agent0_controller.handle_click(event.pos)
                    if hasattr(agent1_controller, 'handle_click'):
                        agent1_controller.handle_click(event.pos)

        if paused:
            pygame.time.wait(50)
            continue

        # Get agent 0 action
        if hasattr(agent0_controller, 'predict'):
            # RL agent
            agent0_action, _ = agent0_controller.predict(obs, deterministic=True)
        elif hasattr(agent0_controller, 'get_current_waypoint'):
            # Human direct controller - set waypoint directly
            waypoint = agent0_controller.get_current_waypoint()
            base_env.agents[base_env.aircraft_ids[0]].waypoint_override = waypoint
            agent0_action = 0  # Dummy action since waypoint is set directly
        elif hasattr(agent0_controller, 'get_current_action'):
            # Human subpolicy controller
            agent0_action = agent0_controller.get_current_action()
        else:
            # Heuristic agent - this is handled differently through the wrapper
            agent0_action = 0

        # Get agent 1 action (handled by wrapper if it's a teammate policy)
        if hasattr(agent1_controller, 'should_override_waypoint'):
            human_action = agent1_controller.get_current_action()
            if agent1_controller.should_override_waypoint():
                custom_waypoint = agent1_controller.get_custom_waypoint()
                custom_waypoint = (int(custom_waypoint[0]), int(custom_waypoint[1]))
                base_env.agents[base_env.aircraft_ids[1]].waypoint_override = custom_waypoint

        # Take step in environment
        obs, reward, done, info = env.step([agent0_action])
        obs = obs[0]
        reward = reward[0]
        done = done[0]
        info = info[0]

        episode_reward += reward
        step_count += 1

        # Render the environment
        env.render()
        pygame.display.flip()
        clock.tick(tick_rate)

        # Print periodic status
        if step_count % 100 == 0:
            print(f"Step {step_count}: Reward = {episode_reward:.2f}, "
                  f"Targets = {info.get('target_ids', -1)}, "
                  f"Threats = {info.get('threat_ids', -1)}")

    print(f"\nEpisode Complete!")
    print(f"Final Reward: {episode_reward:.2f}")
    print(f"Steps Taken: {step_count}")
    print(f"Targets Identified: {info.get('target_ids', -1)}")
    print(f"Threats Identified: {info.get('threat_ids', -1)}")

    return False, episode_reward, step_count


def main():
    """Main function with configurable agent setups"""

    # Configuration - modify these variables to change agent setups
    AGENT0_TYPE = "rl"  # Options: "rl", "heuristic", "human"
    AGENT0_MODEL_PATH = "./experiments/exp2_user_study/saved_agents/aug2b_finetuned_model.zip"
    AGENT0_VECNORM_PATH = "./experiments/exp2_user_study/saved_agents/aug2b_finetuned_vecnormalize.pkl"
    AGENT0_HEURISTIC_TYPE = "conservative"  # Options: "conservative", "aggressive", "coordinated", "greedy"

    AGENT1_TYPE = "human"  # Options: "rl", "heuristic", "human", "none"
    AGENT1_MODEL_PATH = "./exp2_user_study/saved_agents/aug2b_baseline_model.zip"
    AGENT1_VECNORM_PATH = "./exp2_user_study/saved_agents/aug2b_baseline_vecnormalize.pkl"
    AGENT1_HEURISTIC_TYPE = "aggressive"

    NUM_EPISODES = 5
    LEVEL_NUMBER = 1  # Which level to play (1-7)
    TICK_RATE = 30

    # Load configuration
    config_filename = './configs/main_config.json'
    config = load_env_config(config_filename)
    config['tick_rate'] = TICK_RATE
    config['force_specific_level'] = LEVEL_NUMBER - 1
    config['action_type'] = 'Discrete16'

    print(f"Loaded config from {config_filename}")
    print(f"Agent 0: {AGENT0_TYPE}")
    print(f"Agent 1: {AGENT1_TYPE}")

    # Initialize pygame
    if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'user32'):
        ctypes.windll.user32.SetProcessDPIAware()
    pygame.display.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    window_width, window_height = config['window_size'][0], config['window_size'][1]
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("MAISR Play Environment")

    # Create environment
    env_fns = [make_wrapped_env(config, clock, window, 'black', 'play_env') for _ in range(1)]
    env = DummyVecEnv(env_fns)

    # Set up Agent 0
    agent0_controller = None
    if AGENT0_TYPE == "rl":
        # Load RL agent and VecNormalize wrapper
        if not os.path.exists(AGENT0_MODEL_PATH):
            raise FileNotFoundError(f"Agent 0 model not found: {AGENT0_MODEL_PATH}")

        if AGENT0_VECNORM_PATH and os.path.exists(AGENT0_VECNORM_PATH):
            env = load_vecnormalize_wrapper(AGENT0_VECNORM_PATH, env)

        agent0_controller = PPO.load(AGENT0_MODEL_PATH, env=env)
        print(f"Loaded Agent 0 RL model from {AGENT0_MODEL_PATH}")

    elif AGENT0_TYPE == "heuristic":
        # For agent 0 heuristic, we'll need to modify the wrapper to use it
        print(f"Agent 0 will use heuristic policy: {AGENT0_HEURISTIC_TYPE}")
        # Note: This would require modifying the wrapper to handle agent 0 as heuristic
        agent0_controller = None  # Placeholder - would need wrapper modification

    elif AGENT0_TYPE == "human":
        agent0_controller = HumanController(env)
        print("Agent 0 is human-controlled (direct waypoint)")

    # Set up Agent 1 (teammate)
    if AGENT1_TYPE == "rl":
        if not os.path.exists(AGENT1_MODEL_PATH):
            raise FileNotFoundError(f"Agent 1 model not found: {AGENT1_MODEL_PATH}")

        agent1_model = PPO.load(AGENT1_MODEL_PATH)
        # Create RL teammate policy and set it in the wrapper
        from utility.league_management import RLTeammatePolicy

        teammate_policy = RLTeammatePolicy(
            model=agent1_model,
            env=None,
            local_search_policy=None,
            go_to_highvalue_policy=None,
            change_region_subpolicy=None,
            norm_stats_path=AGENT1_VECNORM_PATH if os.path.exists(AGENT1_VECNORM_PATH) else None
        )
        teammate_policy.name = f"RL_Agent1"

        # Set the teammate policy in the wrapper
        env.envs[0].env.teammate_policy = teammate_policy
        env.envs[0].env.current_teammate = teammate_policy

        agent1_controller = None  # Handled by wrapper
        print(f"Loaded Agent 1 RL model from {AGENT1_MODEL_PATH}")

    elif AGENT1_TYPE == "heuristic":
        teammate_policy = create_heuristic_agent(AGENT1_HEURISTIC_TYPE, config)

        # Set the teammate policy in the wrapper
        env.envs[0].env.teammate_policy = teammate_policy
        env.envs[0].env.current_teammate = teammate_policy

        agent1_controller = None  # Handled by wrapper
        print(f"Agent 1 using heuristic policy: {AGENT1_HEURISTIC_TYPE}")

    elif AGENT1_TYPE == "human":
        agent1_controller = HumanController(env)
        print("Agent 1 is human-controlled (direct waypoint)")

    elif AGENT1_TYPE == "none":
        # Disable teammate
        env.envs[0].env.set_teammate_active(False)
        agent1_controller = None
        print("Agent 1 disabled (single agent mode)")

    # Run episodes
    try:
        for episode in range(NUM_EPISODES):
            print(f"\n{'=' * 20} Episode {episode + 1}/{NUM_EPISODES} {'=' * 20}")

            should_quit, episode_reward, step_count = run_episode(
                env, agent0_controller, agent1_controller, config, window, clock, TICK_RATE
            )

            if should_quit:
                print("Experiment terminated by user")
                break

            # Wait between episodes
            if episode < NUM_EPISODES - 1:
                print("Next episode starting in 2 seconds...")
                pygame.time.wait(2000)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()