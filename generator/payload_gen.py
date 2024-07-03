import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from payload_generator_rl.controls.modifier import ModifyPayload
from payload_generator_rl.controls import modifier
from tensorflow.keras.models import load_model
import pickle
import random
from gymnasium import spaces

# Load the detection model and related assets
vec_file = "C:/TEST/generator/xss_detector/file/word2vec.pickle"
model_file = "C:/TEST/generator/xss_detector/file/Conv_model.keras"

# Load the detection model
detection_model = load_model(model_file)
# Load embeddings from word2vec.pickle file
with open(vec_file, "rb") as f:
    word2vec_model = pickle.load(f)
    dictionary = word2vec_model["dictionary"]
    reverse_dictionary = word2vec_model["reverse_dictionary"]
    embeddings = word2vec_model["embeddings"]

from xss_detector.check import check_xss

ACTION_LOOKUP = {i: act for i, act in enumerate(modifier.ACTION_TABLE.keys())}

class XSSMutationEnv(gym.Env):
    def __init__(self, payloads, max_steps=100):
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
        return observation

    def step(self, action):
        action = int(action)

        if self.temp_banned_action is not None and action == self.temp_banned_action:
            available_actions = list(set(range(len(ACTION_LOOKUP))) - {self.temp_banned_action})
            action = random.choice(available_actions)

        action_method_name = ACTION_LOOKUP[action]
        mutation_function = getattr(self.modifier, action_method_name, None)

        if mutation_function:
            try:
                mutated_payloads = mutation_function([self.current_payload])
                mutated_payload = mutated_payloads[0]
            except Exception as e:
                mutated_payload = self.current_payload
        else:
            mutated_payload = self.current_payload

        reward = self._evaluate_mutation(mutated_payload)

        if reward == 20:
            self.current_payload = mutated_payload
            self.temp_banned_action = None
        elif reward == -20:
            self.temp_banned_action = action

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        self.mutation_history[self.current_step - 1] = action

        return self._get_observation(), reward, terminated, truncated, {}

    def _evaluate_mutation(self, payload):
        success = check_xss(payload, dictionary, embeddings, reverse_dictionary, detection_model)
        if success == 1:
            return -20
        elif success == 0:
            return 20

# Read payloads from the text file
with open('C:/Test/generator/dataset/portswigger.txt', 'r') as file:
    payloads = file.read().splitlines()

# Initialize the environment with the payloads from the file
env = XSSMutationEnv(payloads)

# Load the trained PPO agent
model = PPO.load("C:/Test/ppo_model_lstm")

# Generate mutated payloads using the trained PPO model
def generate_mutated_payloads(env, model, num_episodes=100):
    mutated_payloads = []
    for _ in range(num_episodes):
        obs = env.reset()[0]
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            mutated_payloads.append(env.current_payload)
    return mutated_payloads

# Generate payloads
mutated_payloads = generate_mutated_payloads(env, model, num_episodes=100)

# Function to calculate detection and escape rates
def calculate_detection_escape_rates(mutated_payloads):
    total_adversarial_examples = len(mutated_payloads)
    malicious_detected = 0
    benign_detected = 0

    for payload in mutated_payloads:
        success = check_xss(payload, dictionary, embeddings, reverse_dictionary, detection_model)
        if success == 1:
            malicious_detected += 1
        elif success == 0:
            benign_detected += 1

    detection_rate = malicious_detected / total_adversarial_examples
    escape_rate = benign_detected / total_adversarial_examples

    return detection_rate, escape_rate

# Calculate detection and escape rates
detection_rate, escape_rate = calculate_detection_escape_rates(mutated_payloads)

print(f"Detection Rate: {detection_rate * 100:.2f}%")
print(f"Escape Rate: {escape_rate * 100:.2f}%")

print(f"Generated {len(mutated_payloads)} mutated payloads.")
