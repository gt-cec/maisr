import warnings
warnings.filterwarnings("ignore", message="Your system is avx2 capable but pygame was not built with support for it")
import env_test_suite
from utility.data_logging import load_env_config
import cProfile, pstats, io
from pstats import SortKey

if __name__ == "__main__":
    config = load_env_config('../config_files/june9a.json')
    config['eval_freq'] = 4900
    config['n_eval_episodes'] = 5
    config['num_timesteps'] = 1e5

    with cProfile.Profile() as pr:
        env_test_suite.test_env_train(config)

    ps = pstats.Stats(pr).sort_stats(SortKey.CUMULATIVE)
    ps.strip_dirs()
    ps.print_stats()