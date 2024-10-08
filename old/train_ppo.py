import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import isr_env
import save_callback
import os

# set up the log directory
log_dir = "./"
os.makedirs(log_dir, exist_ok=True)

# create the ISR gym
env = isr_env.ISREnv(num_ships=10, num_threat_classes=4)
env = Monitor(env, log_dir)

# define the model
n_steps = 100
batch_size = 10
if ("best_model.zip" in os.listdir("./")):
    model = PPO.load("./best_model.zip", env=env, batch_size=batch_size, verbose=1, n_steps=n_steps)
    print("Loaded existing best_model.zip")
else:
    model = PPO("MultiInputPolicy", env, batch_size=batch_size, verbose=1, n_steps=n_steps)

# set up the callback
check_freq = 1  # save frequency
callback = save_callback.SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=log_dir)

# train the model
model.learn(total_timesteps=10_000, callback=callback)
model.save("./model_ppo_nsteps_" + str(n_steps) + "_batchsize_" + str(batch_size))

# eval the model
vec_env = model.get_env()
obs = vec_env.reset()
print("train ppo: done reset")
for i in range(1000):
    print("train ppo: predicting")
    action, _state = model.predict(obs, deterministic=True)
    print("train ppo: stepping")
    obs, reward, done, info = vec_env.step(action)
    print("train ppo: done step")
    vec_env.render("human")
    print("train ppo: done render")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()