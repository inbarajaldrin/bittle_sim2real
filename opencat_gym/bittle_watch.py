from stable_baselines3 import PPO
from bittle_env import OpenCatGymEnv

def watch_model():
    # Load environment with GUI
    env = OpenCatGymEnv()
    env.GUI_MODE = True
    
    # Load trained model on CPU (matches training device)
    model = PPO.load("./saved_models/opencat_gym_model", device="cpu")
    
    obs = env.reset()[0]  # Gymnasium returns (obs, info)
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs = env.reset()[0]

if __name__ == "__main__":
    watch_model()