import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from generator.xss_detector.check import *

#Code detector 
from tensorflow.keras.models import load_model
# Model path
#model_file = "C:/KLTN/xss_detector/file/Conv_model.keras"
model_file = "C:/KLTN/xss_detector/file/MLP_model.keras"
#model_file = "C:/KLTN/xss_detector/file/LSTM_model.keras"

vec_file = "C:/KLTN/xss_detector/file/word2vec.pickle"
# Load mô hình MLP
model = load_model(model_file)
# Load embeddings từ file word2vec.pickle
with open(vec_file, "rb") as f:
    word2vec_model = pickle.load(f)
    dictionary = word2vec_model["dictionary"]
    reverse_dictionary = word2vec_model["reverse_dictionary"]
    embeddings = word2vec_model["embeddings"]


# Assuming the existing code is saved as a module or script, import it.
from generator.payload_generator_rl.controls.modifier import ModifyPayload
from generator.payload_generator_rl.controls import modifier

ACTION_LOOKUP = {i: act for i, act in enumerate(modifier.ACTION_TABLE.keys())}

class XSSMutationEnv(gym.Env):
    def __init__(self, payloads, max_steps=100):
        super(XSSMutationEnv, self).__init__()

        self.payloads = payloads
        self.max_steps = max_steps

        self.modifier = ModifyPayload()

        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        # Observation space as a single Box space combining current_payload and mutation_history
        self.observation_space = spaces.Box(
            low=0,
            high=max(len(payloads), len(ACTION_LOOKUP)),
            shape=(1 + max_steps,),
            dtype=np.int32
        )

        self.current_payload_index = 0
        self.mutation_history = np.zeros(self.max_steps, dtype=np.int32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.current_payload_index = random.randint(0, len(self.payloads) - 1)
        self.mutation_history = np.zeros(self.max_steps, dtype=np.int32)
        self.current_step = 0

        return self._get_observation(), {}

    def _get_observation(self):
        # Combine current_payload_index and mutation_history into a single array
        return np.concatenate(([self.current_payload_index], self.mutation_history)).astype(np.int32)

    def step(self, action):
        # Ensure action is an integer
        action = int(action)
        current_payload = self.payloads[self.current_payload_index]
        action_method_name = ACTION_LOOKUP[action]
        mutation_function = getattr(self.modifier, action_method_name, None)

        if mutation_function:
            try:
                mutated_payloads = mutation_function([current_payload])
                mutated_payload = mutated_payloads[0]
                self.payloads[self.current_payload_index] = mutated_payload
                print(f"{mutated_payload}")
            except Exception as e:
                print(f"Mutation failed: {e}")
                mutated_payload = current_payload
        else:
            mutated_payload = current_payload

        reward = self._evaluate_mutation(mutated_payload)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # Update this based on your specific environment's logic

        self.mutation_history[self.current_step - 1] = action

        return self._get_observation(), reward, terminated, truncated, {}

    def _evaluate_mutation(self, payload):
        # Placeholder reward function
        success = random.choice([True, False])
        return 10 if success else -1

# Read payloads from the text file
with open('C:\Test\generator\dataset\portswigger.txt', 'r') as file:
    payloads = file.read().splitlines()

# Initialize the environment with the payloads from the file
env = XSSMutationEnv(payloads)

# Load the trained PPO agent
trained_model = PPO.load("C:\Test\generator\ppo_xss_mutation_test.zip")

# Use the agent to generate payloads
obs, _ = env.reset()
done = False
generated_payloads = []

while not done:
    # Directly use the dictionary observation
    action, _states = trained_model.predict(obs.astype(np.int32))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    generated_payloads.append(env.payloads[env.current_payload_index])

print("Generated Payloads:")
for payload in generated_payloads:
    print(payload)

    #Goi nguyen ham nay luon nha Hao. Neu thay doi bien payload thi thay doi "payload" thoi
    check_xss(payload, dictionary, embeddings, reverse_dictionary, model)
