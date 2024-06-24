import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Assuming the existing code is saved as a module or script, import it.
from payload_generator_rl.controls.modifier import ModifyPayload
from payload_generator_rl.controls import modifier

from tensorflow.keras.models import load_model
import pickle

# Load the detection model and related assets
model_file = "C:/TEST/generator/xss_detector/file/MLP_model.keras"
vec_file = "C:/TEST/generator/xss_detector/file/word2vec.pickle"

# Load the detection model
detection_model = load_model(model_file)
# Load embeddings from word2vec.pickle file
with open(vec_file, "rb") as f:
    word2vec_model = pickle.load(f)
    dictionary = word2vec_model["dictionary"]
    reverse_dictionary = word2vec_model["reverse_dictionary"]
    embeddings = word2vec_model["embeddings"]

# Assuming the existing code is saved as a module or script, import it.
from xss_detector.check import check_xss

ACTION_LOOKUP = {i: act for i, act in enumerate(modifier.ACTION_TABLE.keys())}

class XSSMutationEnv(gym.Env):
    def __init__(self, payloads, max_steps=10):
        super(XSSMutationEnv, self).__init__()

        self.payloads = payloads
        self.max_steps = max_steps

        self.modifier = ModifyPayload()

        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.observation_space = spaces.Box(
            low=0,
            high=max(len(payloads), len(ACTION_LOOKUP)),
            shape=(1 + max_steps,),
            dtype=np.int32
        )

        self.current_payload_index = 0
        self.mutation_history = np.zeros(self.max_steps, dtype=np.int32)
        self.current_step = 0
        self.temp_banned_action = None
        self.current_payload = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_payload_index = random.randint(0, len(self.payloads) - 1)
        self.current_payload = self.payloads[self.current_payload_index]
        self.mutation_history = np.zeros(self.max_steps, dtype=np.int32)
        self.current_step = 0
        self.temp_banned_action = None

        return self._get_observation(), {}

    def _get_observation(self):
        observation = np.concatenate(([self.current_payload_index], self.mutation_history)).astype(np.int32)
        print(f"Observation shape: {observation.shape}")
        return observation

    def step(self, action):
        action = int(action)

        if self.temp_banned_action is not None and action == self.temp_banned_action:
            available_actions = list(set(range(len(ACTION_LOOKUP))) - {self.temp_banned_action})
            action = random.choice(available_actions)

        action_method_name = ACTION_LOOKUP[action]
        mutation_function = getattr(self.modifier, action_method_name, None)

        print(f"Action: {action} - {action_method_name}")

        if mutation_function:
            try:
                mutated_payloads = mutation_function([self.current_payload])
                mutated_payload = mutated_payloads[0]
                print(f"Mutated Payload: {mutated_payload}")
            except Exception as e:
                print(f"Mutation failed: {e}")
                mutated_payload = self.current_payload
        else:
            mutated_payload = self.current_payload

        reward = self._evaluate_mutation(mutated_payload)
        print(f"Reward: {reward}")

        if reward == 10:
            self.current_payload = mutated_payload
            self.temp_banned_action = None
        else:
            self.temp_banned_action = action
            print(f"Action {action} is temporarily banned.")

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        self.mutation_history[self.current_step - 1] = action

        return self._get_observation(), reward, terminated, truncated, {}

    def _evaluate_mutation(self, payload):
        success = check_xss(payload, dictionary, embeddings, reverse_dictionary, detection_model)
        print(f"Check XSS success: {success}")
        if success == 1:
            return -1
        elif success == 0:
            return 10

# Read payloads from the text file
with open('C:/Test/generator/dataset/portswigger.txt', 'r') as file:
    payloads = file.read().splitlines()

# Initialize the environment with the payloads from the file
env = XSSMutationEnv(payloads)

# Check the environment
check_env(env, warn=True)

# Initialize the PPO agent with MlpPolicy
model = PPO('MlpPolicy', env, verbose=1)

# Train the PPO agent
model.learn(total_timesteps=10)

# Save the trained model
model.save("ppo_model_test")
