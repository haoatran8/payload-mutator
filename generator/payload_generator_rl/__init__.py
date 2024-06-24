from gym.envs.registration import register
from sklearn.model_selection import train_test_split

MAXTURNS = 50

register(
    id="xssenv-v1",
    entry_point="payload_generator_rl.envs:XSSEnv",
    kwargs={
        "max_turns": MAXTURNS,
    },
)

register(
    id="testenv-v1",
    entry_point="payload_generator_rl.envs:XSSMutationEnv",
)