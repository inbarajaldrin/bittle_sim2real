from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from bittle_env_control import OpenCatGymEnv, MAXIMUM_LENGTH

def train():
    parallel_envs = 8
    
    envs = make_vec_env(
        OpenCatGymEnv,
        n_envs=parallel_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            'randomize_command': True  # Now properly passed to __init__
        }
    )
    
    
    # Network architecture with separate policy/value networks
    custom_arch = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    model = PPO('MlpPolicy', envs,
              seed=42,
              policy_kwargs=custom_arch,
              n_steps=int(2048*8/parallel_envs),  # 2048 steps per env
              verbose=1,
              device="cpu").learn(int(MAXIMUM_LENGTH))

    model.save("./trained_models/opencat_gym_model")

if __name__ == "__main__":
    train()