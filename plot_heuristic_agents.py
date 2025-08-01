import itertools
import os
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from env_multi_new import MAISREnvVec
from training_wrappers.localsearch_training_wrapper import MaisrLocalSearchWrapper
from utility.data_logging import load_env_config
from utility.league_management import TeammateManager, GenericTeammatePolicy, SubPolicy, LocalSearch, ChangeRegions, GoToNearestThreat, TargetSearchLocalTSP, RecordedTrajectoryTeammate


def run_and_plot(env, episodes=7):
    for level in range(episodes):
        env.envs[0].env.config['force_specific_level'] = level
        #obs = env.reset()
        done, truncated = False, False

        while not done:
            action = env.action_space.sample()

            obses, rewards, dones, infos = env.step([action])
            obs = obses[0]
            reward = rewards[0]
            info = infos[0]
            done = dones[0]



def main():
    config_file = "configs/Monolith_index_August.json"
    config = load_env_config(config_file)
    config['teammate_active_at_start'] = True
    config['render_mode'] = 'headless'

    risk_tolerances = ["low", "medium", "high", "max_greedy"]
    spatial_coords = [False, True]
    action_stabilities = ["stable", "noisy", "very_noisy"]

    for risk, coord, stability in itertools.product(risk_tolerances, spatial_coords, action_stabilities):
        combo_name = f"risk_{risk}_coord_{coord}_stability_{stability}"
        print(f"=== Running combo: {combo_name} ===")

        subpolicies = {
            'local_search': LocalSearch(model_path=None),  # Using heuristic
            'change_region': ChangeRegions(model_path=None),  # Using heuristic
            'go_to_threat': GoToNearestThreat(model_path=None),  # Using heuristic
            'local_tsp_nocoord': TargetSearchLocalTSP(search_radius=200),
            'global_tsp_nocoord': TargetSearchLocalTSP(search_radius=1000),

            'local_tsp_yescoord': TargetSearchLocalTSP(search_radius=200, spatial_coord=True),
            'global_tsp_yescoord': TargetSearchLocalTSP(search_radius=1000, spatial_coord=True)

        }

        tm = TeammateManager(
            league_type="strategy_diverse",
            balance_method="uniform",
            subpolicies=subpolicies,
            selfplay_checkpoint_dir=None,
            pretrained_teammate_dir=None,
        )
        tm.overfit_test = None
        teammate = tm._create_strategy_diverse_heuristic_teammate()
        teammate.mode_selector_agent.risk_tolerance = risk
        teammate.mode_selector_agent.spatial_coord = coord
        teammate.action_stability = stability


        def make_env():
            base_env = MAISREnvVec(config=config, render_mode='headless', tag='heuristictests0')

            local_search_policy = LocalSearch()
            go_to_highvalue_policy = GoToNearestThreat(model_path=None)
            change_region_subpolicy = ChangeRegions(model_path=None)
            evade_policy = None

            wrapped_env = MaisrLocalSearchWrapper(
                base_env,
                0.01,
                local_search_policy,
                go_to_highvalue_policy,
                change_region_subpolicy,
                evade_policy,
                teammate_policy=teammate,
                #observation_noise_std = 0.01
            )
            wrapped_env.current_teammate = teammate
            return wrapped_env

        env = DummyVecEnv([make_env])
        os.makedirs(f"plots/{combo_name}", exist_ok=True)
        env.envs[0].env.run_name = f"plots/{combo_name}"

        teammate.env = env.envs[0].env

        run_and_plot(env, episodes=7)
        env.close()


if __name__ == "__main__":
    main()
