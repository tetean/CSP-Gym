from gymnasium.envs.registration import register

register(
    id='CSPEnv-SiC-v0',
    entry_point='environments.SiC:CSPEnvSiC',
    max_episode_steps=100,
)

register(
    id='CSPEnv-Ar-v0',
    entry_point='environments.Ar:CSPEnv',
    max_episode_steps=100,
)