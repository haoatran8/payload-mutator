import gym
import numpy as np
from gym.spaces import Discrete, Box
from collections import OrderedDict

from payload_generator_rl.controls import modifier

ACTION_LOOKUP = {i: act for i, act in enumerate(modifier.ACTION_TABLE.keys())}

class XSSEnv(gym.Env):
    def __init__(self, max_turns=100, original_code=""):
        super(XSSEnv, self).__init__()

        self.action_space = Discrete(len(ACTION_LOOKUP))
        observation_high = np.finfo(np.float32).max
        self.state = np.array([0.0, 0.0])
        self.observation_space = Box(low=-observation_high, high=observation_high, shape=(2381,), dtype=np.float32)

        # Initialize the environments parameters
        self.max_turns = max_turns
        self.history = OrderedDict()
        self.original_code = original_code
        self.current_code = original_code
        self.current_step = 0

    def apply_action(self, action_ix):
        action = ACTION_LOOKUP[action_ix]
        self.history[action] = self.history.get(action, 0) + 1
        modified_code = getattr(modifier, action)([self.current_code])
        self.current_code = modified_code[0]

        return self.current_code

    def step(self, action_ix):
        self.current_step += 1
        # Apply the selected action to the code
        modified_code = self.apply_action(action_ix)

        modified_code = modified_code[:len(self.original_code)].ljust(len(self.original_code), ' ')
        # Calculate the reward (constant reward of 10)
        reward = 10
        # Update the current state
        self.current_code = modified_code

        # Check if the episode is done (optional)
        done = self.current_step >= self.max_turns
    
        # Return the next state, reward, whether the episode is done, and additional information
        return self.get_observation(), reward, done, {}

    def reset(self):
        # Reset the environment to the initial state
        self.current_code = self.original_code
        self.history.clear()
        self.current_step = 0
        return self.get_observation()

    def render(self, mode='human'):
        pass

    #def get_observation(self):
    #    observation = np.zeros(len(self.current_code))
    #    return observation
    
    def get_observation(self):
    # Convert current code to an array of shape (2381,)
    # Assuming you want to pad or truncate `self.current_code` to match the observation space shape.
        code_array = np.fromstring(self.current_code, dtype=np.uint8)[:2381]  # Convert current code to uint8 array
        code_array = np.pad(code_array, (0, max(0, 2381 - len(code_array))), 'constant')  # Pad to length 2381
        return code_array
