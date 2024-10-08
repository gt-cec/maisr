# creates a custom Gym environment using ISR

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import requests
import time

class ISREnv(gym.Env):
    """Custom ISR Environment following the Gym format"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    env_url = "http://localhost:100/"

    def __init__(self, num_ships, num_threat_classes):
        super().__init__()
        # Define action and observation space
        # for ISR, action space is a waypoint (x,y)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # for ISR, observation space is the gameboard state
        self.observation_space = spaces.Dict({
            "agent1 pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "agent2 pos": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            "threat pos": spaces.Box(low=-1, high=1, shape=(num_ships,2), dtype=np.float32),
            "threat class": spaces.MultiBinary([num_ships,num_threat_classes])
        })

    def step(self, action):
        x = float(action[0])
        y = float(action[1])

        # increment the step
        target_step = requests.post(self.env_url + "rl-step", json={"source": "rl", "action": [x, y]}).json()["target step"]

        # get the observation space and check the response parameters
        while True:
            resp = requests.get(self.env_url + "rl-observation").json()
            if "current step" not in resp:
                raise ValueError("Missing 'current step' key from /rl-observation GET")
            if "target step" not in resp:
                raise ValueError("Missing 'target step' key from /rl-observation GET")
            if "observation" not in resp:
                raise ValueError("Missing 'observation' key from /rl-observation GET")
            if "reward" not in resp:
                raise ValueError("Missing 'reward' key from /rl-observation GET")
            if "terminated" not in resp:
                raise ValueError("Missing 'terminated' key from /rl-observation GET")
            if "truncated" not in resp:
                raise ValueError("Missing 'truncated' key from /rl-observation GET")
            # if the step is the target step, break
            if int(resp["current step"]) >= int(target_step):
                # print("Environment has stepped! Current", resp["current step"], "target", resp["target step"])
                break
            # print("Waiting for target step", resp["target step"], target_step, "currently", resp["current step"])
            time.sleep(.05)

        # format the observation into an RL-compatible form
        observation = {
            "agent1 pos": np.array(resp["observation"]["agent1 pos"], dtype=np.float32),
            "agent2 pos": np.array(resp["observation"]["agent2 pos"], dtype=np.float32),
            "threat pos": np.array(resp["observation"]["threat pos"], dtype=np.float32),
            "threat class": np.array(resp["observation"]["threat class"], dtype=np.int8)
        }

        print("Step()", observation["agent1 pos"], observation["agent2 pos"], resp["terminated"])

        info = {}
        return observation, float(resp["reward"]), bool(resp["terminated"]), bool(resp["truncated"]), info

    def reset(self, seed=None, options=None):
        # send the reset command
        resp = requests.post(self.env_url + "rl-reset", json={"command": "init reset"}).json()
        if "result" not in resp or resp["result"] != "success":
            raise ValueError("Response from /rl-reset was not a success! Check the server.")
        print("sent reset")
        # wait until the environment resets
        while True:
            # get the current reset state and check the response parameters
            resp = requests.get(self.env_url + "rl-reset").json()
            if "reset env" not in resp:
                raise ValueError("Missing 'reset env' key from /rl-reset GET")
            if "env is reset" not in resp:
                raise ValueError("Missing 'env is reset' key from /rl-reset GET")
            # exit when the env has reset
            if resp["env is reset"] is True:
                break
            time.sleep(.1)
            print("    waiting", resp)
        print("finished resetting")
        # get the observation space and check the response parameters
        resp = requests.get(self.env_url + "rl-observation").json()
        if "current step" not in resp:
            raise ValueError("Missing 'current step' key from /rl-observation GET")
        if "target step" not in resp:
            raise ValueError("Missing 'target step' key from /rl-observation GET")
        if "observation" not in resp:
            raise ValueError("Missing 'observation' key from /rl-observation GET")
        if resp["current step"] != 0:
            raise ValueError("The environment was just reset, however the 'current step' parameter is not 0! Check the JS handling of reset.")

        # format the observation into an RL-compatible form
        observation = {
            "agent1 pos": np.array(resp["observation"]["agent1 pos"], dtype=np.float32),
            "agent2 pos": np.array(resp["observation"]["agent2 pos"], dtype=np.float32),
            "threat pos": np.array(resp["observation"]["threat pos"], dtype=np.float32),
            "threat class": np.array(resp["observation"]["threat class"], dtype=np.int8)
        }
        print("DONE RESET")

        return observation, {}

    def render(self):
        print("render()")
        return

    def close(self):
        # reset the environment
        resp = requests.post(self.env_url + "rl-reset", json={"command": "init reset"}).json
        if "result" not in resp or resp["result"] != "success":
            raise ValueError("Response from /rl-reset was not a success! Check the server.")
        print("Environment has been closed")
        return
    

if __name__ == "__main__":
    # when we run this file, we check the environment
    env = ISREnv(num_ships=10, num_threat_classes=4)
    check_env(env)
