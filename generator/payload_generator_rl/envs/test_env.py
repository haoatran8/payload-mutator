import gym
from gym import spaces
import numpy as np
import random

from payload_generator_rl.controls.modifier import ModifyPayload
from payload_generator_rl.controls import modifier

ACTION_LOOKUP = {i: act for i, act in enumerate(modifier.ACTION_TABLE.keys())}

class XSSMutationEnv(gym.Env):
    def __init__(self, payloads, max_steps=100):
        super(XSSMutationEnv, self).__init__()
        
        self.payloads = payloads
        self.max_steps = max_steps
        
        self.modifier = ModifyPayload()
        
        self.action_space = spaces.Discrete(len(ACTION_LOOKUP))
        self.observation_space = spaces.Dict({
            "current_payload": spaces.Discrete(len(payloads)),
            "mutation_history": spaces.Box(low=0, high=len(ACTION_LOOKUP), shape=(max_steps,), dtype=np.int32)
        })
        
        self.current_payload_index = 0
        self.mutation_history = []
        self.current_step = 0

    def reset(self):
        self.current_payload_index = random.randint(0, len(self.payloads) - 1)
        self.mutation_history = np.zeros(self.max_steps, dtype=np.int32)
        self.current_step = 0
        
        return self._get_observation()

    def _get_observation(self):
        return {
            "current_payload": self.current_payload_index,
            "mutation_history": self.mutation_history
        }

    def step(self, action):
        current_payload = self.payloads[self.current_payload_index]
        action_method_name = ACTION_LOOKUP[action]
        mutation_function = getattr(self.modifier, action_method_name, None)

        if mutation_function:
            try:
                mutated_payloads = mutation_function([current_payload])
                mutated_payload = mutated_payloads[0]
                self.payloads[self.current_payload_index] = mutated_payload
                # print(f"Action: {action_method_name}, Original Payload: {current_payload}, Mutated Payload: {mutated_payload}")
                print(f"{mutated_payload}")
            except Exception as e:
                print(f"Mutation failed: {e}")
                mutated_payload = current_payload
        else:
            mutated_payload = current_payload

        reward = self._evaluate_mutation(mutated_payload)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        self.mutation_history[self.current_step - 1] = action
        
        return self._get_observation(), reward, done, {}

    def _evaluate_mutation(self, payload):
        # Placeholder reward function
        success = random.choice([True, False])
        return 10 if success else -1
    
