import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


# Constants to define training and visualisation.
GUI_MODE = True          # Set "True" to display pybullet in a window
EPISODE_LENGTH = 250      # Number of steps for one training episode
MAXIMUM_LENGTH = 2e6    # Number of total steps for entire training

# Factors to weight rewards and penalties.
PENALTY_STEPS = 2e6       # Increase of penalty by step_counter/PENALTY_STEPS
FAC_MOVEMENT = 1000       # Reward movement in x-direction
FAC_STABILITY = 0.1       # Punish body roll and pitch velocities
FAC_Z_VELOCITY = 0.0      # Punish z movement of body
FAC_SLIP = 0.0            # Punish slipping of paws
FAC_ARM_CONTACT = 0.01    # Punish crawling on arms and elbows
FAC_SMOOTH_1 = 1.0        # Punish jitter and vibrational movement, 1st order
FAC_SMOOTH_2 = 1.0        # Punish jitter and vibrational movement, 2nd order
FAC_CLEARANCE = 0.0       # Factor to enfore foot clearance to PAW_Z_TARGET
PAW_Z_TARGET = 0.005      # Target height (m) of paw during swing phase

BOUND_ANG = 110         # Joint maximum angle (deg)
STEP_ANGLE = 11           # Maximum angle (deg) delta per step
ANG_FACTOR = 0.1          # Improve angular velocity resolution before clip.

# Values for randomization, to improve sim to real transfer.
RANDOM_GYRO = 0           # Percent
RANDOM_JOINT_ANGS = 0      # Percent
RANDOM_MASS = 0           # Percent, currently inactive
RANDOM_FRICTION = 0       # Percent, currently inactive

LENGTH_RECENT_ANGLES = 3  # Buffer to read recent joint angles
LENGTH_JOINT_HISTORY = 30 # Number of steps to store joint angles.

# Add to constants
CMD_VEL_DIM = 3  # vx, vy, wz
CMD_NORM = np.array([1.0, 0.5, 2.0], dtype=np.float32)   # vx, vy, wz scale

# Size of oberservation space is set up of: 
# [LENGTH_JOINT_HISTORY, quaternion, gyro]
SIZE_OBSERVATION = LENGTH_JOINT_HISTORY * 8 + 6 + CMD_VEL_DIM  # Added CMD_VEL_DIM



class OpenCatGymEnv(gym.Env):
    """ Gymnasium environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, randomize_command=False):  
        self.randomize_command = randomize_command
        self.step_counter = 0
        self.step_counter_session = 0
        self.state_history = np.array([])
        self.angle_history = np.array([])
        self.bound_ang = np.deg2rad(BOUND_ANG)

        if GUI_MODE:
            p.connect(p.GUI)
            # Uncommend to create a video.
            #video_options = ("--width=960 --height=540 
            #                + "--mp4=\"training.mp4\" --mp4fps=60")
            #p.connect(p.GUI, options=video_options) 
        else:
            # Use for training without visualisation (significantly faster).
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, 
                                     cameraYaw=-170, 
                                     cameraPitch=-40, 
                                     cameraTargetPosition=[0.4,0,0])

        # The action space are the 8 joint angles.
        self.action_space = gym.spaces.Box(np.array([-1]*8), np.array([1]*8))

        # Add command velocity storage
        self.target_velocity = np.zeros(3)  # vx, vy, wz
        self.actual_velocity = np.zeros(3)
        
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, 
            shape=(SIZE_OBSERVATION,),  # Now includes command velocity
            dtype=np.float32
        )
            
    def _get_obs(self):
        """Build observation = [robot state, joint history, cmd_vel]."""
        cmd_norm = self.target_velocity / CMD_NORM
        return np.hstack((self.state_robot, self.angle_history, cmd_norm))
    
    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        joint_angs = np.asarray(p.getJointStates(self.robot_id, self.joint_id),
                                                   dtype=object)[:,0]
        ds = np.deg2rad(STEP_ANGLE) # Maximum change of angle per step
        joint_angs += action * ds # Change per step including agent action

        # Apply joint boundaries individually.
        min_ang = -self.bound_ang
        max_ang = self.bound_ang
        joint_angs[0] = np.clip(joint_angs[0], min_ang, max_ang) # shoulder_left
        joint_angs[1] = np.clip(joint_angs[1], min_ang, max_ang) # elbow_left
        joint_angs[2] = np.clip(joint_angs[2], min_ang, max_ang) # shoulder_right
        joint_angs[3] = np.clip(joint_angs[3], min_ang, max_ang) # elbow_right
        joint_angs[4] = np.clip(joint_angs[4], min_ang, max_ang) # hip_right
        joint_angs[5] = np.clip(joint_angs[5], min_ang, max_ang) # knee_right
        joint_angs[6] = np.clip(joint_angs[6], min_ang, max_ang) # hip_left
        joint_angs[7] = np.clip(joint_angs[7], min_ang, max_ang) # knee_left

        # Transform angle to degree and perform rounding, because 
        # OpenCat robot have only integer values.
        joint_angsDeg = np.rad2deg(joint_angs.astype(np.float64))
        joint_angsDegRounded = joint_angsDeg.round()
        joint_angs = np.deg2rad(joint_angsDegRounded)

        # Simulate delay for data transfer. Delay has to be modeled to close 
        # "reality gap").
        p.stepSimulation()

        # Check for friction of paws, to prevent slipping while training.
        paw_contact = []
        paw_idx = [3, 6, 9, 12]
        for idx in paw_idx:
            paw_contact.append(True if p.getContactPoints(bodyA=self.robot_id, 
                                                          linkIndexA=idx) 
                                    else False)

        paw_slipping = 0
        for in_contact in np.nonzero(paw_contact)[0]:
            paw_slipping += np.linalg.norm((
                            p.getLinkState(self.robot_id,
                                           linkIndex=paw_idx[in_contact], 
                                           computeLinkVelocity=1)[0][0:1]))

        # Read clearance of paw from ground
        paw_clearance = 0
        for idx in paw_idx:
            paw_z_pos = p.getLinkState(self.robot_id, linkIndex=idx)[0][2]
            paw_clearance += (paw_z_pos-PAW_Z_TARGET)**2 * np.linalg.norm(
                (p.getLinkState(self.robot_id, linkIndex=idx, 
                                computeLinkVelocity=1)[0][0:1]))**0.5

        # Check if elbows or lower arm are in contact with ground
        arm_idx = [1, 2, 4, 5]
        for idx in arm_idx:
            if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx):
                self.arm_contact += 1

        # Read clearance of torso from ground
        base_clearance = p.getBasePositionAndOrientation(self.robot_id)[0][2]

        # Set new joint angles
        p.setJointMotorControlArray(self.robot_id, 
                                    self.joint_id, 
                                    p.POSITION_CONTROL, 
                                    joint_angs, 
                                    forces=np.ones(8)*0.2)
        p.stepSimulation() # Delay of data transfer

        # Normalize joint_angs
        joint_angs[0] /= self.bound_ang
        joint_angs[1] /= self.bound_ang
        joint_angs[2] /= self.bound_ang
        joint_angs[3] /= self.bound_ang
        joint_angs[4] /= self.bound_ang
        joint_angs[5] /= self.bound_ang
        joint_angs[6] /= self.bound_ang
        joint_angs[7] /= self.bound_ang

        # Adding every 2nd angle to the joint angle history.
        if(self.step_counter % 2 == 0):
            self.angle_history = np.append(self.angle_history, 
                                           self.randomize(joint_angs, 
                                                          RANDOM_JOINT_ANGS))
            self.angle_history = np.delete(self.angle_history, np.s_[0:8])

        self.recent_angles = np.append(self.recent_angles, joint_angs)
        self.recent_angles = np.delete(self.recent_angles, np.s_[0:8])

        joint_angs_prev = self.recent_angles[8:16]
        joint_angs_prev_prev = self.recent_angles[0:8]

        # Read robot state (pitch, roll and their derivatives of the torso).
        state_pos, state_ang = p.getBasePositionAndOrientation(self.robot_id)
        p.stepSimulation() # Emulated delay of data transfer via serial port
        state_ang_euler = np.asarray(p.getEulerFromQuaternion(state_ang)[0:2])
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        state_vel = state_vel[0:2]*ANG_FACTOR
        state_vel_clip = np.clip(state_vel, -1, 1)
        self.state_robot = np.concatenate((state_ang, state_vel_clip))
        current_position = p.getBasePositionAndOrientation(self.robot_id)[0][0] 

        #  joint‑angle smoothness  (Jerk and acceleration penalties)
        smooth_movement = np.sum(
            FAC_SMOOTH_1 * np.square(joint_angs - joint_angs_prev) +
            FAC_SMOOTH_2 * np.square(joint_angs - 2*joint_angs_prev + joint_angs_prev_prev)
        )

        #  body angular‑velocity stability
        z_velocity = p.getBaseVelocity(self.robot_id)[0][2]
        body_stability = (
            FAC_STABILITY * (state_vel_clip[0]**2 + state_vel_clip[1]**2)
            + FAC_Z_VELOCITY * z_velocity**2
        )
        # ---------------------------------------------------------------

        # ---------- NEW reward block -------------------------------------------
        # (1) basic smoothness & stability you already computed
        penalty_basic = smooth_movement + body_stability

        # (2) velocity tracking --------------------------------------------------
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        v_actual  = np.array([lin_vel[0], lin_vel[1], ang_vel[2]])
        v_target  = self.target_velocity

        # normalise so each component is roughly in [-1, 1]
        v_actual_n = v_actual / CMD_NORM
        v_target_n = v_target / CMD_NORM
        v_error_n  = v_target_n - v_actual_n

        align       = np.dot(v_target_n, v_actual_n)      # ∈ [-1, +1]
        speed_cost  = np.sum(v_error_n ** 2)              # ∈ [0, 9] roughly

        # weights (tweak later)
        W_ALIGN   = 30.0
        W_SPEED   = 10.0
        W_PENALTY =  1.0

        # final scalar reward
        reward = (
            W_ALIGN   * align           # (+) follow command direction
        - W_SPEED   * speed_cost      # (–) magnitude error
        - W_PENALTY * penalty_basic   # (–) jitter & body motion
        )

        # ---------- termination logic -------------------------------------------
        terminated = truncated = False
        self.step_counter += 1
        if self.step_counter > EPISODE_LENGTH:
            truncated = True
        elif self.is_fallen():
            terminated = True

        # ---------- build observation & return ----------------------------------
        self.observation = self._get_obs()
        return (
            self.observation.astype(np.float32),
            reward,
            terminated,
            truncated,
            {},          # info dict (empty for now)
        )


    def reset(self, seed=None, options=None):
        # Generate random command velocity if enabled
        if self.randomize_command:
            self.target_velocity = np.array([
                np.random.uniform(-0.5, 1.0),  # vx
                np.random.uniform(-0.2, 0.2),  # vy
                np.random.uniform(-1.0, 1.0)   # wz
            ])
        else:
            self.target_velocity = np.zeros(3)
        
        self.step_counter = 0
        self.arm_contact = 0
        p.resetSimulation()
        # Disable rendering during loading.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) 
        p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")

        start_pos = [0,0,0.08]
        start_orient = p.getQuaternionFromEuler([0,0,0])

        urdf_path = "models/"#"/content/drive/My Drive/opencat-gym-esp32/models/"
        self.robot_id = p.loadURDF(urdf_path + "bittle_esp32.urdf", 
                                   start_pos, start_orient, 
                                   flags=p.URDF_USE_SELF_COLLISION) 
        
        # Initialize urdf links and joints.
        self.joint_id = []
        #paramIds = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            joint_name = info[1]
            joint_type = info[2]

            if (joint_type == p.JOINT_PRISMATIC 
                or joint_type == p.JOINT_REVOLUTE):
                self.joint_id.append(j)
                #paramIds.append(p.addUserDebugParameter(joint_name.decode("utf-8")))
                # Limiting motor dynamics. Although bittle's dynamics seem to 
                # be be quite high like up to 7 rad/s.
                p.changeDynamics(self.robot_id, j, maxJointVelocity = np.pi*10) 
        
        # Setting start position. This influences training.
        joint_angs = np.deg2rad(np.array([1, 0, 1, 0, 1, 0, 1, 0])*50) 

        i = 0
        for j in self.joint_id:
            p.resetJointState(self.robot_id,j, joint_angs[i])
            i = i+1

        # Normalize joint angles.
        joint_angs[0] /= self.bound_ang
        joint_angs[1] /= self.bound_ang
        joint_angs[2] /= self.bound_ang
        joint_angs[3] /= self.bound_ang
        joint_angs[4] /= self.bound_ang
        joint_angs[5] /= self.bound_ang
        joint_angs[6] /= self.bound_ang
        joint_angs[7] /= self.bound_ang

        # Read robot state (pitch, roll and their derivatives of the torso)
        state_ang = p.getBasePositionAndOrientation(self.robot_id)[1]
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        state_vel = state_vel[0:2]*ANG_FACTOR
        self.state_robot = np.concatenate((state_ang, 
                                           np.clip(state_vel, -1, 1)))

        # Initialize robot state history with reset position
        state_joints = np.asarray(
            p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:,0]
        state_joints /= self.bound_ang 
        
        self.angle_history = np.tile(state_joints, LENGTH_JOINT_HISTORY)
        self.recent_angles = np.tile(state_joints, LENGTH_RECENT_ANGLES)
        self.observation = self._get_obs()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        info = {}
        return self.observation.astype(np.float32), info

    def set_target_velocity(self, vx, vy, wz):
        self.target_velocity = np.array([vx, vy, wz])

    def render(self, mode='human'):
        pass


    def close(self):
        p.disconnect()


    def is_fallen(self):
        """ Check if robot is fallen. It becomes "True", 
            when pitch or roll is more than 1.3 rad.
        """
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        orient = p.getEulerFromQuaternion(orient)
        is_fallen = (np.fabs(orient[0]) > 1.3 
                    or np.fabs(orient[1]) > 1.3)

        return is_fallen


    def randomize(self, value, percentage):
        """ Randomize value within percentage boundaries.
        """
        percentage /= 100
        value_randomized = value * (1 + percentage*(2*np.random.rand()-1))

        return value_randomized