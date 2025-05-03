import tkinter as tk
from tkinter import ttk
from stable_baselines3 import PPO
from bittle_env_control import OpenCatGymEnv

class VelocityControlApp:
    def __init__(self, master, env, model):
        self.master = master
        self.env = env
        self.model = model
        
        # Initialize velocity commands
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        
        # Setup GUI
        self.setup_gui()
        
        # Initialize simulation
        self.obs = self.env.reset()[0]
        self.running = True
        self.update_simulation()  # Start the simulation loop

    def setup_gui(self):
        self.master.title("Bittle Velocity Control")
        
        # Velocity X control
        ttk.Label(self.master, text="Linear Velocity X (m/s)").pack()
        self.vx_slider = ttk.Scale(self.master, from_=-1.0, to=1.0, 
                                 command=lambda v: self.set_velocity('vx', v))
        self.vx_slider.set(0)
        self.vx_slider.pack()

        # Velocity Y control
        ttk.Label(self.master, text="Linear Velocity Y (m/s)").pack()
        self.vy_slider = ttk.Scale(self.master, from_=-0.5, to=0.5,
                                 command=lambda v: self.set_velocity('vy', v))
        self.vy_slider.set(0)
        self.vy_slider.pack()

        # Angular Velocity Z control
        ttk.Label(self.master, text="Angular Velocity Z (rad/s)").pack()
        self.wz_slider = ttk.Scale(self.master, from_=-2.0, to=2.0,
                                 command=lambda v: self.set_velocity('wz', v))
        self.wz_slider.set(0)
        self.wz_slider.pack()

        # Quit button
        ttk.Button(self.master, text="Quit", command=self.quit).pack()

    def set_velocity(self, axis, value):
        """Update target velocity from slider input"""
        value = float(value)
        if axis == 'vx':
            self.vx = value
        elif axis == 'vy':
            self.vy = value
        elif axis == 'wz':
            self.wz = value

    def update_simulation(self):
        """Main simulation update loop"""
        if self.running:
            # Set target velocity in environment
            self.env.set_target_velocity(self.vx, self.vy, self.wz)
            
            # Predict action from model
            action, _ = self.model.predict(self.obs, deterministic=True)
            
            # Step environment
            self.obs, _, terminated, truncated, _ = self.env.step(action)
            
            # Reset if episode done
            if terminated or truncated:
                self.obs = self.env.reset()[0]
            
            # Schedule next update
            self.master.after(50, self.update_simulation)

    def quit(self):
        """Cleanup and exit"""
        self.running = False
        self.env.close()
        self.master.destroy()

def watch_model():
    # Initialize environment and model
    env = OpenCatGymEnv()
    model = PPO.load("./trained_models/opencat_gym_model", device="cpu")
    
    # Create Tkinter interface
    root = tk.Tk()
    app = VelocityControlApp(root, env, model)
    root.mainloop()

if __name__ == "__main__":
    watch_model()