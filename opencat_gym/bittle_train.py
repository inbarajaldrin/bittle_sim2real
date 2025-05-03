from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from bittle_env import OpenCatGymEnv, MAXIMUM_LENGTH  # Import the constant

def train():
    parallel_envs = 8
    envs = make_vec_env(OpenCatGymEnv, 
                      n_envs=parallel_envs, 
                      vec_env_cls=SubprocVecEnv)

    custom_arch = dict(net_arch=[256, 256])
    
    model = PPO('MlpPolicy', envs,
              seed=42,
              policy_kwargs=custom_arch,
              n_steps=int(2048*8/parallel_envs),
              verbose=1,
              device="cpu").learn(int(MAXIMUM_LENGTH))  # Convert to int

    model.save("./trained_models/opencat_gym_model")

if __name__ == "__main__":
    train()