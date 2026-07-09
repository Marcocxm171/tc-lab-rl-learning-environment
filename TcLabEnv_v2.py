import numpy as np
import gymnasium as gym
from gymnasium import spaces
 
# defining the gym environment class
class TCLabEnv(gym.Env):
 
    def __init__(self):
 
        super().__init__()

                # initialise the state to ambient temp 
        self.Ta = 21 + 273.15 # ambient temp 
        self.T_heater = self.Ta # initialise the heater temp - hidden state for update
        self.T_sensor = self.Ta  # initialise sensor's temp - observed state for agent
        self.t = 0

        self._P1 = 200

        self.previous_Q = 0.0 # store previous heater action for aggressive control cost

        self.setpoint = 60 + 273.15 # initialise the setpoint for 60C
        self.safety_limit = 100 + 273.15


        # setting some integration settings
        self.euler_step = 0.2
        self.time_step = 1
        self.max_time = 300
 
        # defining the observation space
        # env cannot have a temperature lower than ambient as we don't have a cooler
        self.observation_space = spaces.Box(
            low=np.array([self.Ta - 273.15], dtype=np.float32),
            high=np.array([120.0], dtype=np.float32),
            dtype=np.float32
        )
 
        # defining the action space
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([100.0], dtype=np.float32),
            dtype=np.float32
        )


 
    def reset(self, seed=None, options=None):
        #resetting the environment back to the start at room temp
        super().reset(seed=seed)
 
        self.T_heater = self.Ta
        self.T_sensor = self.Ta
        self.t = 0.0

        self.previous_Q = 0.0 # reset previous heater action

        # we will randomise the setpoint for cycle - to train a more robust agent (NOT yet)
        self.setpoint = 60 + 273.15

        observation = np.array([self.T_sensor - 273.15], dtype=np.float32)
        info = {}
 
        return observation, info
 
 
    def step (self,action):
        # we figure out the next temperature state based on the current temperature and the current heating power
       
        # convert action into a scalar
        Q = float(np.asarray(action, dtype=np.float32).item())
        Q = float(np.clip(Q, 0.0, 100.0))

        for _ in range(int(self.time_step / self.euler_step)):

            dT_heater_dt = self._P1 * Q / 5720 + (self.Ta - self.T_heater) / 16.6
            dT_sensor_dt = (self.T_heater - self.T_sensor) / 140

            self.T_heater += dT_heater_dt * self.euler_step
            self.T_sensor += dT_sensor_dt * self.euler_step
                # time update
        
        self.t = self.t + self.time_step

        # need to calculate the reward for agent
        error_cost = abs(self.T_sensor - self.setpoint) / 10

        # penalise using too much heater power
        power_cost = 0.01 * (Q / 100.0) ** 2

        # penalise aggressive changes in heater power
        delta_Q = Q - self.previous_Q
        aggressive_cost = 0.05 * (delta_Q / 100.0) ** 2

        # safety penalty
        safety_cost = 0.0
        if self.T_sensor >= self.safety_limit:
            safety_cost = 10.0

        cost = error_cost + power_cost + aggressive_cost + safety_cost
        reward = - cost

        # update previous action after calculating reward
        self.previous_Q = Q
 
 
        # info - show additional information that the agent is allowed to see but not typically used
        info = {
            "T_K": self.T_sensor,
            "T_C": self.T_sensor - 273.15,
            "Q_percent": Q,
            "time": self.t,
            "error_cost": error_cost,
            "power_cost": power_cost,
            "aggressive_cost": aggressive_cost,
        }
 
        # episode ends when the maximum time is reached
        truncated = self.t >= self.max_time

        terminated = self.T_sensor >= self.safety_limit
       
        observation = np.array(
            [float(self.T_sensor - 273.15)],
            dtype=np.float32
        )
       
       
        return observation, reward, terminated, truncated, info